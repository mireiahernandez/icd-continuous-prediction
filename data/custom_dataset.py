import pandas as pd
import numpy as np
import torch
import tqdm as tqdm
from tqdm import tqdm
import ast
import os
import itertools
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """Custom dataset for icd-9 code prediction.

    Code based on HTDC (Ng et al, 2022)"""

    def __init__(
        self,
        notes_agg_df,
        tokenizer,
        max_chunks,
        setup="latest",  # 'uniform
        limit_ds=0,
        batch_size=None,
    ):
        self.notes_agg_df = notes_agg_df
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        self.setup = setup
        self.limit_ds = limit_ds
        np.random.seed(1)

    def __len__(self):
        return len(self.notes_agg_df)

    def tokenize(self, doc):
        return self.tokenizer(
            doc,
            truncation=True,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

    def _get_note_end_chunk_ids(self, seq_ids):
        id = seq_ids[0]
        change_points = []
        for i, seq_id in enumerate(seq_ids):
            if seq_id != id:
                change_points.append(i - 1)
                id = seq_id
        # append last index, as it is always the indication of the last note
        change_points.append(i)
        return change_points

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

    def filter_mask(self, seq_ids):
        """Get selected indices according to the logic:
        1. All indices of the first note
        2. All indices of the last note (a.k.a. discharge summary))
        3. Randomly select the remaining indices from the middle notes"""
        first_indices = np.where(seq_ids == seq_ids[0])[0]
        # limit DS to 4 chunks
        last_indices = np.where(seq_ids == seq_ids[-1])[0]
        # limit last indices if more than max_chunks - len(first_indices)
        # selecting the last max_chunks - len(first_indices) indices
        last_indices = last_indices[
            -min(len(last_indices), self.max_chunks - len(first_indices)) :
        ]
        last_indices = last_indices[-self.limit_ds :]
        middle_indices = np.where(
            np.logical_and(seq_ids > seq_ids[0], seq_ids < seq_ids[-1])
        )[0]
        middle_indices = np.sort(
            np.random.choice(
                middle_indices,
                max(
                    0,
                    min(
                        len(middle_indices),
                        self.max_chunks - len(first_indices) - len(last_indices),
                    ),
                ),
                replace=False,
            )
        )
        return first_indices.tolist() + middle_indices.tolist() + last_indices.tolist()

    def __getitem__(self, idx):
        data = self.notes_agg_df.iloc[idx]

        output = [self.tokenize(doc) for doc in data.TEXT]
        hadm_id = data.HADM_ID
        # doc[input_ids] is (# chunks, 512), i.e., if note is longer than 512, it returns len/512 # chunks
        input_ids = torch.cat(
            [doc["input_ids"] for doc in output]
        )  # this concatenates to (overall # chunks, 512)
        attention_mask = torch.cat([doc["attention_mask"] for doc in output])
        seq_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [[i] * len(output[i]["input_ids"]) for i in range(len(output))]
                )
            )
        )
        category_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.CATEGORY_INDEX[i]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )
        hours_elapsed = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [data.HOURS_ELAPSED[i]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )

        percent_elapsed = data.PERCENT_ELAPSED
        label = torch.FloatTensor(data.ICD9_CODE_BINARY)

        input_ids = input_ids[:]
        attention_mask = attention_mask[:]
        seq_ids = torch.LongTensor(seq_ids)
        category_ids = torch.LongTensor(category_ids)
        seq_ids = seq_ids[:]
        category_ids = category_ids[:]

        # if latest setup, select last max_chunks
        if self.setup == "latest":
            input_ids = input_ids[-self.max_chunks :]
            attention_mask = attention_mask[-self.max_chunks :]
            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)
            seq_ids = seq_ids[-self.max_chunks :]
            category_ids = category_ids[-self.max_chunks :]
            hours_elapsed = hours_elapsed[-self.max_chunks :]

        # in a uniform setting, select first and last note
        # and randomly sample the rest
        elif self.setup == "uniform":
            indices_mask = self.filter_mask(np.array(seq_ids))
            input_ids = input_ids[indices_mask]
            attention_mask = attention_mask[indices_mask]
            seq_ids = seq_ids[indices_mask]
            category_ids = category_ids[indices_mask]
            hours_elapsed = hours_elapsed[indices_mask]
        
        elif self.setup == "random":
            # keep all notes and random sample during training
            # while keeping all notes for inference
            pass 
        
        else:
            raise ValueError("Invalid setup")

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
            "label": label,
            "hadm_id": hadm_id,
            "hours_elapsed": hours_elapsed,
            "cutoffs": cutoffs,
        }
