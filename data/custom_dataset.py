
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
    """ Custom dataset for icd-9 code prediction.
    
    Code based on HTDC (Ng et al, 2022)"""
    def __init__(
        self,
        notes_agg_df,
        tokenizer,
        max_chunks,
        priority_mode="Last",  # "First, "Last", "Index", "Diverse"
        priority_idxs=[],
        batch_size=None,
        filter_time='all',
    ):
        self.notes_agg_df = notes_agg_df
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        assert priority_mode in ["First", "Last", "Index", "None", "Diverse"]
        self.priority_mode = priority_mode
        self.priority_idxs = priority_idxs
        self.filter_time = filter_time

        if self.priority_mode == "Index":
            self.priority_scores_dict = {i: 0 for i in range(15)}
            # Prioritise Discharge Summaries (idx = 5) first, followed by selected idx
            self.priority_scores_dict[5] = 2
            for idx in self.priority_idxs:
                self.priority_scores_dict[idx] = 1
        # self.samples = [self._prepare_data_sample(i) for i in tqdm(range(len(self.notes_agg_df)))]

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
        cutoffs = {'2d': -1, '6d': -1, '14d': -1, 'noDS': -1, 'all': -1}
        for i, (hour, cat) in enumerate(zip(hours_elapsed, category_ids)):
            if cat != 5:
                if hour < 2*24:
                    cutoffs['2d'] = i
                if hour < 6*24:
                    cutoffs['6d'] = i
                if hour < 14*24:
                    cutoffs['14d'] = i
                cutoffs['noDS'] = i
            # cutoffs['all'] = i
        return cutoffs
    

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

        # create dictionary of temporal points

        if self.priority_mode == "Index":
            priority_scores = np.vectorize(self.priority_scores_dict.get)(category_ids)
            priority_indices = np.argsort(priority_scores, kind="stable")
            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)

            input_ids = input_ids[priority_indices]
            attention_mask = attention_mask[priority_indices]
            seq_ids = seq_ids[priority_indices]
            category_ids = category_ids[priority_indices]

        if self.priority_mode == "Diverse":
            category_reverse_seqids = np.array(
                list(
                    itertools.chain.from_iterable(
                        [
                            [data.CATEGORY_REVERSE_SEQID[i]]
                            * len(output[i]["input_ids"])
                            for i in range(len(output))
                        ]
                    )
                )
            )
            priority_indices = np.argsort(
                -np.array(category_reverse_seqids), kind="stable"
            )

            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)

            input_ids = input_ids[priority_indices]
            attention_mask = attention_mask[priority_indices]
            seq_ids = seq_ids[priority_indices]
            category_ids = category_ids[priority_indices]

        if self.priority_mode == "First":
            input_ids = input_ids[: self.max_chunks]
            attention_mask = attention_mask[: self.max_chunks]
            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)
            seq_ids = seq_ids[: self.max_chunks]
            category_ids = category_ids[: self.max_chunks]
        
        if self.priority_mode == "None":
            input_ids = input_ids[:]
            attention_mask = attention_mask[:]
            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)
            seq_ids = seq_ids[:]
            category_ids = category_ids[:]

        else:
            input_ids = input_ids[-self.max_chunks :]
            attention_mask = attention_mask[-self.max_chunks :]
            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)
            seq_ids = seq_ids[-self.max_chunks :]
            category_ids = category_ids[-self.max_chunks :]
        
        # store the final chunk of each note
        # note_end_chunk_ids = self._get_note_end_chunk_ids(seq_ids)
        # print(f"Seq ids before: {seq_ids}")

        seq_id_vals = torch.unique(seq_ids).tolist()
        seq_id_dict = {seq: idx for idx, seq in enumerate(seq_id_vals)}
        seq_ids = seq_ids.apply_(seq_id_dict.get)
        # print(f"Seq ids after: {seq_ids}")
        cutoffs = self._get_cutoffs(hours_elapsed, category_ids)
        # print(cutoffs)
        # get last 16 chunks until the cutoff
        if self.filter_time != 'all':
            assert self.filter_time in ['2d', '6d', '14d', 'noDS']
            assert self.priority_mode == 'None'
            cutoff = cutoffs[self.filter_time]
            first_chunk = max(0, cutoff-self.max_chunks+1)
            input_ids = input_ids[first_chunk : cutoff+1]

            attention_mask = attention_mask[first_chunk : cutoff+1]
            seq_ids = torch.LongTensor(seq_ids)
            category_ids = torch.LongTensor(category_ids)
            seq_ids = seq_ids[first_chunk : cutoff+1]
            category_ids = category_ids[first_chunk : cutoff+1]
        

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_ids": seq_ids,
            "category_ids": category_ids,
            # "note_end_chunk_ids": note_end_chunk_ids,
            "label": label,
            "hadm_id": hadm_id,
            "hours_elapsed": hours_elapsed,
            "cutoffs": cutoffs
        }