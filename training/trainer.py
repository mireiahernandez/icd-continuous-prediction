import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
import tqdm as tqdm
import torch.optim as optim
import ast
import os
import itertools
from torch.utils.data import Dataset
import torch.utils.checkpoint
import random
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from evaluation.metrics import MyMetrics
from evaluation.evaluate import evaluate
from data.custom_dataset import OneSampleDataset
import ipdb
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
)


class Trainer:
    """ Custom trainer for icd-9 code prediction.
    
    Code based on HTDC (Ng et al, 2022)"""
    def __init__(
        self,
        model,
        optimizer,
        scaler,
        lr_scheduler,
        config,
        device,
        dtype
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.dtype = dtype
    
    def select_documents(self, encoding_grads):
        """ Select documents with largest gradients.
        Args:
            encoding_grads: gradients of the token encodings (shape: ((Nc x T) x D))
        Returns:
            document_mask: mask of documents to keep (shape: (Nc, T))
        """
        # Calculate gradient norms of encoding_grads along dimension 1 (i.e. along dimension D)
        # shape (NcxT)
        gradient_norms = torch.norm(encoding_grads, dim=1)
        # Reshape gradient_norms to shape (Nc, T)
        gradient_norms = gradient_norms.reshape((-1, 512))
        # Calculate the mean gradient norm for each document
        mean_gradient_norms = torch.mean(gradient_norms, dim=1) # shape (Nc)
        # Select the top k documents with the largest mean gradient norm
        # shape (self.config["max_chunks"])
        topk = torch.topk(mean_gradient_norms, self.config["max_chunks"])
        # Create a mask of documents to keep
        # shape (Nc, T)
        document_mask = torch.zeros_like(gradient_norms, dtype=torch.float32).cpu()
        # Set the top k documents to 1
        document_mask[topk.indices, :] = 1
        # # Reshape document_mask to shape (NcxT, D)
        # document_mask = document_mask.reshape((-1))
        # # Expand document_mask to shape (NcxT, D)
        # document_mask = document_mask[:, None].expand_as(encoding_grads)
        return document_mask
    

    def train(
        self,
        training_generator,
        training_args,
        validation_generator,
        grad_accumulation_steps=1,
        epochs=1,
    ):
        self.model = self.model.to(device=self.device)  # move the model parameters to CPU/GPU
        self.model.train()  # put model to training mode
        mymetrics = MyMetrics(debug=self.config["debug"])
        print("evaluate temporal is ", self.config["evaluate_temporal"])
        for e in range(training_args['TOTAL_COMPLETED_EPOCHS'], epochs):
            hyps = []
            refs = []
            if self.config["evaluate_temporal"]:
                hyps_temp ={'2d':[],'5d':[],'13d':[],'noDS':[]}
                refs_temp ={'2d':[],'5d':[],'13d':[],'noDS':[]}
            for t, sample in enumerate(tqdm(training_generator)):
                #--------> First pass through the model: obtain largest gradients
                encoding_grads = []
                one_sample_dataset = OneSampleDataset(sample)
                one_sample_dataloader = DataLoader(one_sample_dataset, batch_size=self.config["max_chunks"], shuffle=False)

                for j, data in enumerate(one_sample_dataloader):
                    labels = data["label"][0,0,:50]
                    input_ids = data["input_ids"]
                    attention_mask = data["attention_mask"]
                    seq_ids = data["seq_ids"]
                    category_ids = data["category_ids"]
                    # note_end_chunk_ids = data["note_end_chunk_ids"]
                    cutoffs = data["cutoffs"]
                    with torch.cuda.amp.autocast(enabled=True) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
                    # with autocast():
                    # with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        scores, token_encodings = self.model(
                            input_ids=input_ids.to(self.device, dtype=torch.long),
                            attention_mask=attention_mask.to(self.device, dtype=torch.long),
                            seq_ids=seq_ids.to(self.device, dtype=torch.long),
                            category_ids=category_ids.to(self.device, dtype=torch.long),
                            # note_end_chunk_ids=note_end_chunk_ids,
                            gradient_selection=True,
                        )
                        loss = F.binary_cross_entropy_with_logits(
                            scores[-1, :][None, :], labels.to(self.device, dtype=self.dtype)[None, :]
                        )

                        self.scaler.scale(loss).backward()
                        #**** GRADIENT SELECTION PART ****#
                        encoding_grads.append(token_encodings.grad)

                        # erase gradients
                        self.optimizer.zero_grad()
                #--------> Select documents with largest gradients
                # concatenate encoding_grads along dimension 0 (i.e. along dimension NcxT)
                encoding_grads = torch.cat(encoding_grads, dim=0)
                document_mask = self.select_documents(encoding_grads)
                # Second pass through the model: obtain predictions
                # apply mask obtained through gradient selection
                labels = sample["label"][0][: self.model.num_labels]
                input_ids = sample["input_ids"][0] # Nc x 512
                attention_mask = sample["attention_mask"][0] # Nc x 512
                seq_ids = sample["seq_ids"][0] # Nc 
                category_ids = sample["category_ids"][0] # Nc
                # note_end_chunk_ids = data["note_end_chunk_ids"]
                cutoffs = sample["cutoffs"]

                # apply document mask to input_ids, attention_mask, seq_ids, category_ids
                # retaining the same shape (2-dim)
                input_ids = input_ids[document_mask.bool()].reshape((-1, 512))
                attention_mask = attention_mask[document_mask.bool()].reshape((-1, 512))
                seq_ids = seq_ids[document_mask.bool()[:,0]] # seq ids is 1-dim
                category_ids = category_ids[document_mask.bool()[:,0]] # category ids is 1-dim

                with torch.cuda.amp.autocast(enabled=True) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
                # with autocast():
                    scores, token_encodings = self.model(
                        input_ids=input_ids.to(self.device, dtype=torch.long),
                        attention_mask=attention_mask.to(self.device, dtype=torch.long),
                        seq_ids=seq_ids.to(self.device, dtype=torch.long),
                        category_ids=category_ids.to(self.device, dtype=torch.long),
                        # note_end_chunk_ids=note_end_chunk_ids,
                    )
                    loss = F.binary_cross_entropy_with_logits(
                        scores[-1, :][None, :], labels.to(self.device, dtype=self.dtype)[None, :]
                    )
                    self.scaler.scale(loss).backward()
                    # convert to probabilities
                    probs = F.sigmoid(scores)
                    # print(f"probs shape: {probs.shape}")
                    # print(f"cutoffs: {cutoffs}")
                    hyps.append(probs[-1, :].detach().cpu().numpy())
                    refs.append(labels.detach().cpu().numpy())

                    if self.config["evaluate_temporal"]:
                        cutoff_times = ['2d','5d','13d','noDS']
                        for time in cutoff_times:
                            if cutoffs[time] != -1:
                                hyps_temp[time].append(probs[cutoffs[time][0], :].detach().cpu().numpy())
                                refs_temp[time].append(labels.detach().cpu().numpy())

                    # print(f"ccutoffs: {cutoffs}, hyprs_temp: {hyps_temp}")
                    if ((t + 1) % grad_accumulation_steps == 0) or (
                        t + 1 == len(training_generator)
                    ):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                # if t==1000:
                #     break
            print("Starting evaluation...")
            print("Epoch: %d" % (training_args['TOTAL_COMPLETED_EPOCHS']))
            if self.config["evaluate_temporal"]:
                 result = self.evaluate_and_save_results(
                    hyps, refs, mymetrics, training_args, validation_generator, hyps_temp, refs_temp
                )
            else:
                result = self.evaluate_and_save_results(
                    hyps, refs, mymetrics, training_args, validation_generator
                ) 

            training_args['CURRENT_PATIENCE_COUNT'] += 1
            training_args['TOTAL_COMPLETED_EPOCHS'] += 1

            if result["validation_f1_micro"] > training_args['CURRENT_BEST']:
                training_args['CURRENT_BEST'] = result['validation_f1_micro']
                training_args['CURRENT_PATIENCE_COUNT'] = 0
                best_path = os.path.join(self.config['project_path'], f"results/BEST_{self.config['run_name']}.pth")
                if self.config["save_model"]:
                    self.save_model(result, training_args, best_path)

            if self.config["save_model"]:
                model_path = os.path.join(
                    self.config['project_path'], f"results/{self.config['run_name']}.pth"
                )
                self._save_model(result, training_args, model_path)

            if (self.config["patience_threshold"] > 0) and (
                training_args['CURRENT_PATIENCE_COUNT'] >= self.config["patience_threshold"]
            ):
                print("Stopped upon hitting early patience threshold ")
                break

            if (self.config["max_epochs"] > 0) and (
                training_args['TOTAL_COMPLETED_EPOCHS'] >= self.config["max_epochs"]
            ):
                print("Stopped upon hitting max number of training epochs")
                break

    def __save_model(self, result, training_args, save_path):
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "results": result,
                    "config": self.config,
                    "epochs": training_args['TOTAL_COMPLETED_EPOCHS'],
                    "current_best": training_args['CURRENT_BEST'],
                    "current_patience_count": training_args['CURRENT_PATIENCE_COUNT'],
                },
                save_path,
            )
    def save_results(self, train_metrics, validation_metrics, training_args, timeframe='all'):
        a = {
            f"validation_{key}": validation_metrics[key]
            for key in validation_metrics.keys()
        }
        b = {f"train_{key}": train_metrics[key] for key in train_metrics.keys()}
        result = {**a, **b}

        #print(result)

        print(
            {
                k: result[k] if type(result[k]) != np.ndarray else {}
                for k in result.keys()
            }
        )
        result["epoch"] = training_args['TOTAL_COMPLETED_EPOCHS']
        result["curr_lr"] = self.lr_scheduler.get_last_lr()
        result.update(self.config)  # add config fields
        result_list = {k: [v] for k, v in result.items()}
        df = pd.DataFrame.from_dict(result_list)  # convert to datframe

        results_path = os.path.join(self.config['project_path'], f"results/{self.config['run_name']}_{timeframe}.csv")
        results_df = pd.read_csv(results_path)
        results_df = pd.concat((results_df, df), axis=0, ignore_index=True)
        results_df.to_csv(results_path)  # update results

        return result

    def evaluate_and_save_results(self, hyps, refs, mymetrics, training_args, validation_generator, hyps_temp=None, refs_temp=None):
        train_metrics = mymetrics.from_numpy(np.asarray(hyps), np.asarray(refs))
        cutoff_times = ['2d', '5d', '13d', 'noDS']
        if self.config["evaluate_temporal"]:
            train_metrics_temp = {time: mymetrics.from_numpy(np.asarray(hyps_temp[time]), np.asarray(refs_temp[time])) for time in cutoff_times}
        #print(train_metrics_temp)
        print(f"Calculating validation metrics with a val dataset of {len(validation_generator)}...")
        validation_metrics, validation_metrics_temp = evaluate(
            mymetrics, self.model, validation_generator, self.device, evaluate_temporal=self.config["evaluate_temporal"], optimise_threshold=True
        )
        #print(validation_metrics_temp)
        result = self.save_results(train_metrics, validation_metrics, training_args, timeframe='all')
        print(result)
        if self.config["evaluate_temporal"]:
            for time in cutoff_times:
                _ = self.save_results(train_metrics_temp[time], validation_metrics_temp[time], training_args, timeframe=time)

        self.model.train()  # put model to training mode


        return result