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
import ipdb
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
)


class Trainer:
    """Custom trainer for icd-9 code prediction.

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
        num_categories,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.dtype = dtype
        # set all key, value pairs in config as attributes
        for key, value in config.items():
            setattr(self, key, value)
        
        self.CosineLoss = nn.CosineEmbeddingLoss(reduction="mean")

    def _get_cutoffs(self, hours_elapsed, category_ids):
        cutoffs = {"2d": -1, "5d": -1, "13d": -1, "noDS": -1, "all": -1}
        for i, (hour, cat) in enumerate(zip(hours_elapsed, category_ids)):
            if cat != 5:
                if hour < 2 * 24:
                    cutoffs["2d"] = i
                if hour < 5 * 24:
                    cutoffs["5d"] = i
                if hour < 13 * 24:
                    cutoffs["13d"] = i
                cutoffs["noDS"] = i
            # cutoffs['all'] = i
        return cutoffs

    def random_sample_sequence(self, data):
        """ Construct a sequence of max_chunks"""
        labels = data["label"][0][: self.model.num_labels]
        input_ids = data["input_ids"][0]
        attention_mask = data["attention_mask"][0]
        seq_ids = data["seq_ids"][0]
        category_ids = data["category_ids"][0]
        hours_elapsed = data["hours_elapsed"][0]

        # select at random 16 indices
        num_idxs = input_ids.shape[0]
        indices_mask = np.arange(num_idxs)
        indices_mask = np.sort(np.random.choice(
            indices_mask,
            min(num_idxs, self.max_chunks),
            replace=False
        ))

        # filter input_ids, attention_mask, seq_ids, category_ids, hours_elapsed
        input_ids = input_ids[indices_mask]
        attention_mask = attention_mask[indices_mask]
        seq_ids = seq_ids[indices_mask]
        category_ids = category_ids[indices_mask]
        hours_elapsed = hours_elapsed[indices_mask]
        # recalculate seq ids based on filtered indices
        seq_id_vals = torch.unique(seq_ids).tolist()
        seq_id_dict = {seq: idx for idx, seq in enumerate(seq_id_vals)}
        seq_ids = seq_ids.apply_(seq_id_dict.get)
        cutoffs = self._get_cutoffs(hours_elapsed, category_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_ids": seq_ids,
            "category_ids": category_ids,
            "labels": labels,
            "hours_elapsed": hours_elapsed,
            "cutoffs": cutoffs,
        }


    def train(
        self,
        training_generator,
        training_args,
        validation_generator,
        grad_accumulation_steps=1,
        epochs=1,
    ):
        self.model = self.model.to(
            device=self.device
        )  # move the model parameters to CPU/GPU
        self.model.train()  # put model to training mode
        mymetrics = MyMetrics(debug=self.config["debug"])
        print("evaluate temporal is ", self.config["evaluate_temporal"])
        for e in range(training_args["TOTAL_COMPLETED_EPOCHS"], epochs):
            preds = {"hyps": [], "refs": [], "hyps_aux": [], "refs_aux": []}
            # add cls, aux, total keys to preds
            train_loss = {"loss_cls": [], "loss_aux": [], "loss_total": []}
            if self.config["evaluate_temporal"]:
                preds["hyps_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}
                preds["refs_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}
            for t, data in enumerate(tqdm(training_generator)):
                aug_data = self.random_sample_sequence(data)
                labels = aug_data["labels"]
                input_ids = aug_data["input_ids"]
                attention_mask = aug_data["attention_mask"]
                seq_ids = aug_data["seq_ids"]
                category_ids = aug_data["category_ids"]
                cutoffs = aug_data["cutoffs"]

                with torch.cuda.amp.autocast(
                    enabled=True
                ) as autocast, torch.backends.cuda.sdp_kernel(
                    enable_flash=False
                ) as disable:
                    # with autocast():
                    scores, doc_embeddings, aux_predictions = self.model(
                        input_ids=input_ids.to(self.device, dtype=torch.long),
                        attention_mask=attention_mask.to(self.device, dtype=torch.long),
                        seq_ids=seq_ids.to(self.device, dtype=torch.long),
                        category_ids=category_ids.to(self.device, dtype=torch.long),
                        # note_end_chunk_ids=note_end_chunk_ids,
                    )
                    # Auxiliary task of predicting next document category
                    if self.config["aux_task"] == "next_document_category":
                        if len(category_ids) > 1 and aux_predictions is not None:
                            true_categories = F.one_hot(
                                # remove 1st category id and add a dummy category end
                                torch.concat(
                                    [
                                        category_ids[1:],
                                        torch.tensor([self.config["num_categories"]]),
                                    ]
                                ),
                                num_classes=self.config["num_categories"] + 1,
                            )
                            loss_aux = F.cross_entropy(
                                aux_predictions,
                                true_categories.to(self.device, dtype=self.dtype),
                            )
                            preds["hyps_aux"].append(aux_predictions.detach().cpu().numpy())
                            preds["refs_aux"].append(true_categories.detach().cpu().numpy())
                    # Auxiliary task of predicting next document embedding
                    elif self.config["aux_task"] == "next_document_embedding":                    
                        # loss is the cosine similarity between the predicted and true next embeddings
                        if len(doc_embeddings) > 1 and aux_predictions is not None:
                            true_embeddings = doc_embeddings[1:]
                            loss_aux = self.CosineLoss(
                                aux_predictions[:-1], # last prediction is not used
                                true_embeddings.to(self.device, dtype=self.dtype).detach(), # detach target
                                torch.ones(true_embeddings.shape[0], device=self.device), # all are positive samples
                            )
                    elif self.config["aux_task"] == "last_document_embedding":
                        # loss is the cosine similarity between the predicted and the LAST true embedding
                        if len(doc_embeddings) > 1 and aux_predictions is not None:
                            true_embeddings = doc_embeddings[-1].repeat(len(doc_embeddings)-1, 1)
                            loss_aux = self.CosineLoss(
                                aux_predictions[:-1], # last prediction is not used
                                true_embeddings.to(self.device, dtype=self.dtype).detach(), # detach target
                                torch.ones(true_embeddings.shape[0], device=self.device), # all are positive samples
                            )
                    else:
                        loss_aux = torch.tensor(0)
                    loss_cls = F.binary_cross_entropy_with_logits(
                        scores[-1, :][None, :],
                        labels.to(self.device, dtype=self.dtype)[None, :],
                    )
                    print(loss_aux)
                    loss = (1 - self.config["weight_aux"]) * loss_cls + self.config[
                        "weight_aux"
                    ] * loss_aux
                    self.scaler.scale(loss).backward()
                    train_loss["loss_cls"].append(loss_cls.detach().cpu().numpy())
                    train_loss["loss_aux"].append(loss_aux.detach().cpu().numpy())
                    train_loss["loss_total"].append(loss.detach().cpu().numpy())
                    # convert to probabilities
                    probs = F.sigmoid(scores)
                    # print(f"probs shape: {probs.shape}")
                    # print(f"cutoffs: {cutoffs}")
                    preds["hyps"].append(probs[-1, :].detach().cpu().numpy())
                    preds["refs"].append(labels.detach().cpu().numpy())
                    if self.config["evaluate_temporal"]:
                        cutoff_times = ["2d", "5d", "13d", "noDS"]
                        for time in cutoff_times:
                            if cutoffs[time] != -1:
                                preds["hyps_temp"][time].append(
                                    probs[cutoffs[time], :].detach().cpu().numpy()
                                )
                                preds["refs_temp"][time].append(
                                    labels.detach().cpu().numpy()
                                )

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
            print("Epoch: %d" % (training_args["TOTAL_COMPLETED_EPOCHS"]))
            result = self.evaluate_and_save_results(
                preds, train_loss, mymetrics, training_args, validation_generator
            )

            training_args["CURRENT_PATIENCE_COUNT"] += 1
            training_args["TOTAL_COMPLETED_EPOCHS"] += 1

            if result["validation_f1_micro"] > training_args["CURRENT_BEST"]:
                training_args["CURRENT_BEST"] = result["validation_f1_micro"]
                training_args["CURRENT_PATIENCE_COUNT"] = 0
                best_path = os.path.join(
                    self.config["project_path"],
                    f"results/BEST_{self.config['run_name']}.pth",
                )
                if self.config["save_model"]:
                    self.save_model(result, training_args, best_path)

            if self.config["save_model"]:
                model_path = os.path.join(
                    self.config["project_path"],
                    f"results/{self.config['run_name']}.pth",
                )
                self._save_model(result, training_args, model_path)

            if (self.config["patience_threshold"] > 0) and (
                training_args["CURRENT_PATIENCE_COUNT"]
                >= self.config["patience_threshold"]
            ):
                print("Stopped upon hitting early patience threshold ")
                break

            if (self.config["max_epochs"] > 0) and (
                training_args["TOTAL_COMPLETED_EPOCHS"] >= self.config["max_epochs"]
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
                "epochs": training_args["TOTAL_COMPLETED_EPOCHS"],
                "current_best": training_args["CURRENT_BEST"],
                "current_patience_count": training_args["CURRENT_PATIENCE_COUNT"],
            },
            save_path,
        )

    def save_results(
        self, train_metrics, validation_metrics, training_args, timeframe="all"
    ):
        """Save resulting metrics (train and val) to csv.
        The argument timeframe specifies the time frame used for evaluation."""
        a = {
            f"validation_{key}": validation_metrics[key]
            for key in validation_metrics.keys()
        }
        b = {f"train_{key}": train_metrics[key] for key in train_metrics.keys()}
        result = {**a, **b}

        # print(result)

        print(
            {
                k: result[k] if type(result[k]) != np.ndarray else {}
                for k in result.keys()
            }
        )
        result["epoch"] = training_args["TOTAL_COMPLETED_EPOCHS"]
        result["curr_lr"] = self.lr_scheduler.get_last_lr()
        result.update(self.config)  # add config fields
        result_list = {k: [v] for k, v in result.items()}
        df = pd.DataFrame.from_dict(result_list)  # convert to datframe

        results_path = os.path.join(
            self.config["project_path"],
            f"results/{self.config['run_name']}_{timeframe}.csv",
        )
        results_df = pd.read_csv(results_path)
        results_df = pd.concat((results_df, df), axis=0, ignore_index=True)
        results_df.to_csv(results_path)  # update results

        return result

    def evaluate_and_save_results(
        self, preds, train_loss, mymetrics, training_args, validation_generator
    ):
        """Evaluate model on validation set and save results to csv."""
        train_metrics = mymetrics.from_numpy(
            np.asarray(preds["hyps"]), np.asarray(preds["refs"])
        )
        # if we have stored some aux predictions (case of next document category prediction)
        if self.config["aux_task"] == "next_document_category":
            train_metrics_aux = mymetrics.from_numpy(
                np.concatenate(preds["hyps_aux"]), np.concatenate(preds["refs_aux"])
            )
        cutoff_times = ["2d", "5d", "13d", "noDS"]
        if self.config["evaluate_temporal"]:
            train_metrics_temp = {
                time: mymetrics.from_numpy(
                    np.asarray(preds["hyps_temp"][time]),
                    np.asarray(preds["refs_temp"][time]),
                )
                for time in cutoff_times
            }
        # print(train_metrics_temp)
        print(
            f"Calculating validation metrics with a val dataset of {len(validation_generator)}..."
        )
        # TODO: set optimise_threshold to True
        val_metrics, val_metrics_temp, val_metrics_aux = evaluate(
            mymetrics,
            self.model,
            validation_generator,
            self.device,
            evaluate_temporal=self.config["evaluate_temporal"],
            optimise_threshold=False,
            num_categories=self.config["num_categories"],
            is_baseline=self.config["is_baseline"],
            aux_task=self.config["aux_task"],
        )
        # print(validation_metrics_temp)
        train_metrics["loss"] = np.mean(train_loss["loss_total"])
        train_metrics["loss_aux"] = np.mean(train_loss["loss_aux"])
        train_metrics["loss_cls"] = np.mean(train_loss["loss_cls"])
        result = self.save_results(
            train_metrics, val_metrics, training_args, timeframe="all"
        )
        # save results of aux task (only if there are some hyps and preds)
        if self.config["aux_task"] == "next_document_category":
            _ = self.save_results(
                train_metrics_aux, val_metrics_aux, training_args, timeframe="all_aux"
            )
        print(result)
        # save results of temp task
        if self.config["evaluate_temporal"]:
            for time in cutoff_times:
                _ = self.save_results(
                    train_metrics_temp[time],
                    val_metrics_temp[time],
                    training_args,
                    timeframe=time,
                )

        self.model.train()  # put model to training mode

        return result



