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
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
)

class Trainer:
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
        
    def train(
        self,
        training_generator,
        training_args,
        grad_accumulation_steps=1,
        epochs=1,
    ):
        self.model = self.model.to(device=self.device)  # move the model parameters to CPU/GPU
        self.model.train()  # put model to training mode
        mymetrics = MyMetrics()

        for e in range(training_args['TOTAL_COMPLETED_EPOCHS'], epochs):
            hyps = []
            refs = []
            for t, data in enumerate(tqdm(training_generator)):
                labels = data["label"][0][: self.model.num_labels]
                input_ids = data["input_ids"][0]
                attention_mask = data["attention_mask"][0]
                seq_ids = data["seq_ids"][0]
                category_ids = data["category_ids"][0]
                note_end_chunk_ids = data["note_end_chunk_ids"]

                # with torch.cuda.amp.autocast(enabled=True) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
                with autocast():
                    scores = self.model(
                        input_ids=input_ids.to(self.device, dtype=torch.long),
                        attention_mask=attention_mask.to(self.device, dtype=torch.long),
                        seq_ids=seq_ids.to(self.device, dtype=torch.long),
                        category_ids=category_ids.to(self.device, dtype=torch.long),
                        note_end_chunk_ids=note_end_chunk_ids,
                    )

                    breakpoint()
                    loss = F.binary_cross_entropy(
                        scores[-1, :][None, :], labels.to(self.device, dtype=self.dtype)[None, :]
                    )
                    self.scaler.scale(loss).backward()
                    hyps.append(scores[-1, :].detach().cpu().numpy())
                    refs.append(labels.detach().cpu().numpy())

                    if ((t + 1) % grad_accumulation_steps == 0) or (
                        t + 1 == len(training_generator)
                    ):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()

            print("Epoch: %d" % (training_args['TOTAL_COMPLETED_EPOCHS']))

            train_metrics = mymetrics.from_numpy(np.asarray(hyps), np.asarray(refs))

            validation_metrics = evaluate(
                mymetrics, self.model, split="validation", optimise_threshold=True
            )

            a = {
                f"validation_{key}": validation_metrics[key]
                for key in validation_metrics.keys()
            }
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

            results_path = os.path.join(self.config['project_path'], f"results/{self.config['run_name']}.csv")
            results_df = pd.read_csv(results_path)
            results_df = results_df.append(df, ignore_index=True)
            results_df.to_csv(results_path)  # update results

            training_args['CURRENT_PATIENCE_COUNT'] += 1
            training_args['TOTAL_COMPLETED_EPOCHS'] += 1

            if result["validation_f1_micro"] > training_args['CURRENT_BEST']:
                PATH = os.path.join(self.config['project_path'], f"results/BEST_{self.config['run_name']}.pth")
                if self.config["save_model"]:
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
                        PATH,
                    )

            PATH = os.path.join(self.config['project_path'], f"results/{self.config['run_name']}.pth")
            if self.config["save_model"]:
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
                    PATH,
                )

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
