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
import ipdb
from evaluation.evaluate import evaluate
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
        dtype,
        categories_mapping
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.dtype = dtype
        self.categories_mapping = categories_mapping
    
    def find_last_indices_with_value_type(self, floats, cat_types):
        # Initialize variables to keep track of the last indices
        last_index_2 = -1
        last_index_5 = -1
        last_index_13 = -1

        # Iterate through the list of floats and value_types
        # 5 -> discharge summary
        for i, (num, value_type) in enumerate(zip(floats, cat_types)):
            if value_type != self.categories_mapping['Discharge summary']:
                if num < 2*24:
                    last_index_2 = i
                if num < 5*24:
                    last_index_5 = i
                if num < 13*24:
                    last_index_13 = i

        # Return the last indices meeting the conditions or -1 if no such indices exist
        return last_index_2, last_index_5, last_index_13 

    def update_hyps_temp(
        self,
        hyps_temp,
        refs_temp,
        probs,
        refs,
        hours_elapsed,
        category_ids,
    ):
        cutoff_2d, cutoff_5d, cutoff_13d = self.find_last_indices_with_value_type(hours_elapsed[0].tolist(), category_ids.tolist())
        try:
            if cutoff_2d != -1:
                hyps_temp['2d'].append(probs[cutoff_2d,:].detach().cpu().numpy())
                refs_temp['2d'].append(refs)
            if cutoff_5d != -1:
                hyps_temp['5d'].append(probs[cutoff_5d,:].detach().cpu().numpy())
                refs_temp['5d'].append(refs)
            if cutoff_13d != -1:
                hyps_temp['13d'].append(probs[cutoff_13d,:].detach().cpu().numpy())
                refs_temp['13d'].append(refs)
        except:
            ipdb.set_trace()
        return hyps_temp, refs_temp

    def validate_loop(self, validation_generator):
        self.model.eval()
        with torch.no_grad():
            ids = []
            hyps = []
            refs = []
            hyps_temp = {'2d': [], '5d': [],'13d': []}
            refs_temp = {'2d': [], '5d': [],'13d': []}
            avail_doc_count = []
            print(f"Starting validation loop...")
            for t, data in enumerate(tqdm(validation_generator)):
                labels = data["label"][0][: self.model.num_labels]

                input_ids = data["input_ids"][0]
                attention_mask = data["attention_mask"][0]
                seq_ids = data["seq_ids"][0]
                category_ids = data["category_ids"][0]
                avail_docs = seq_ids.max().item() + 1
                note_end_chunk_ids = data["note_end_chunk_ids"]
                hours_elapsed = data["hours_elapsed"]
                
                # Nn, L
                scores = self.model(
                    input_ids=input_ids.to(self.device, dtype=torch.long),
                    attention_mask=attention_mask.to(self.device, dtype=torch.long),
                    seq_ids=seq_ids.to(self.device, dtype=torch.long),
                    category_ids=category_ids.to(self.device, dtype=torch.long),
                    note_end_chunk_ids=note_end_chunk_ids,
                )
                probs = F.sigmoid(scores)
                ids.append(data["hadm_id"][0].item())
                avail_doc_count.append(avail_docs)
                hyps.append(probs[-1, :].detach().cpu().numpy())
                hyps_temp, refs_temp = self.update_hyps_temp(hyps_temp, refs_temp, probs, labels.detach().cpu().numpy(), hours_elapsed, category_ids)
                refs.append(labels.detach().cpu().numpy())
        return hyps, hyps_temp, refs, refs_temp

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
        mymetrics = MyMetrics()

        for e in range(training_args['TOTAL_COMPLETED_EPOCHS'], epochs):
            hyps = []
            refs = []
            hyps_temp = {'2d': [], '5d': [],'13d': []}
            refs_temp = {'2d': [], '5d': [],'13d': []}
            for t, data in enumerate(tqdm(training_generator)):
                labels = data["label"][0][: self.model.num_labels]
                input_ids = data["input_ids"][0]
                attention_mask = data["attention_mask"][0]
                seq_ids = data["seq_ids"][0]
                category_ids = data["category_ids"][0]
                note_end_chunk_ids = data["note_end_chunk_ids"]
                hours_elapsed = data["hours_elapsed"]
                hours_elapsed[0][0] = 0 # for DEBUGGING ONLY
                with torch.cuda.amp.autocast(enabled=True) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
                # with autocast():
                    scores = self.model(
                        input_ids=input_ids.to(self.device, dtype=torch.long),
                        attention_mask=attention_mask.to(self.device, dtype=torch.long),
                        seq_ids=seq_ids.to(self.device, dtype=torch.long),
                        category_ids=category_ids.to(self.device, dtype=torch.long),
                        note_end_chunk_ids=note_end_chunk_ids,
                    )

                    loss = F.binary_cross_entropy_with_logits(
                        scores[-1, :][None, :], labels.to(self.device, dtype=self.dtype)[None, :]
                    )
                    self.scaler.scale(loss).backward()
                    # convert to probabilities

                    probs = F.sigmoid(scores)
                    hyps.append(probs[-1, :].detach().cpu().numpy())
                    refs.append(labels.detach().cpu().numpy())
                    hyps_temp, refs_temp = self.update_hyps_temp(hyps_temp, refs_temp, probs, labels.detach().cpu().numpy(), hours_elapsed, category_ids)

                    if ((t + 1) % grad_accumulation_steps == 0) or (
                        t + 1 == len(training_generator)
                    ):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()

            print("Starting evaluation...")
            print("Epoch: %d" % (training_args['TOTAL_COMPLETED_EPOCHS']))
            
            val_hyps, val_hyps_temp, val_refs, val_refs_temp = self.validate_loop(validation_generator)
            result = self.compute_and_save_results(hyps, refs, val_hyps, val_refs, mymetrics, training_args)
            if hyps_temp['2d'] != []:
                result_2d = self.compute_and_save_results(hyps_temp['2d'], refs_temp['2d'], val_hyps_temp['2d'], val_refs_temp['2d'], mymetrics, training_args, result_type='2d')
            if hyps_temp['5d'] != []:
                result_5d = self.compute_and_save_results(hyps_temp['5d'], refs_temp['5d'], val_hyps_temp['5d'], val_refs_temp['5d'], mymetrics, training_args, result_type='5d')
            if hyps_temp['13d'] != []:
                result_13d = self.compute_and_save_results(hyps_temp['13d'], refs_temp['13d'], val_hyps_temp['13d'], val_refs_temp['13d'], mymetrics, training_args, result_type='13d')

            training_args['CURRENT_PATIENCE_COUNT'] += 1
            training_args['TOTAL_COMPLETED_EPOCHS'] += 1

            if result["validation_f1_micro"] > training_args['CURRENT_BEST']:
                CURRENT_BEST = result['validation_f1_micro']
                CURRENT_PATIENCE_COUNT = 0
                best_path = os.path.join(self.config['project_path'], f"results/BEST_{self.config['run_name']}.pth")
                if self.config["save_model"]:
                    self._save_model(result, training_args, best_path)

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



    def _save_model(self, result, training_args, save_path):
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
    
    def compute_and_save_results(
            self, 
            hyps,
            refs,
            val_hyps,
            val_refs,
            mymetrics,
            training_args,
            result_type = 'all'
        ):
        train_metrics = mymetrics.from_numpy(np.asarray(hyps), np.asarray(refs))
        validation_metrics = mymetrics.from_numpy(np.asarray(val_hyps), np.asarray(val_refs))

        a = {f"validation_{key}": validation_metrics[key] for key in validation_metrics.keys()}
        b = {f"train_{key}": train_metrics[key] for key in train_metrics.keys()}
        result = {**a, **b}
        print(result)
        self.model.train()  # put model to training mode

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

        results_path = os.path.join(self.config['project_path'], f"results/{self.config['run_name']}_{result_type}.csv")
        results_df = pd.read_csv(results_path)
        results_df = pd.concat((results_df, df), axis=0, ignore_index=True)
        results_df.to_csv(results_path)  # update results

        return result