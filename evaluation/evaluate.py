from tqdm import tqdm
import torch
import pandas as pd
import os
import numpy as np


def evaluate(
    mymetrics,
    model,
    generator,
    device,
    pred_cutoff=0.5,
    optimise_threshold=False,
):
    model.eval()
    with torch.no_grad():
        ids = []
        hyps = []
        refs = []
        avail_doc_count = []
        print(f"Starting validation loop...")
        for t, data in enumerate(tqdm(generator)):
            labels = data["label"][0][: model.num_labels]

            input_ids = data["input_ids"][0]
            attention_mask = data["attention_mask"][0]
            seq_ids = data["seq_ids"][0]
            category_ids = data["category_ids"][0]
            avail_docs = seq_ids.max().item() + 1
            note_end_chunk_ids = data["note_end_chunk_ids"]

            scores = model(
                input_ids=input_ids.to(device, dtype=torch.long),
                attention_mask=attention_mask.to(device, dtype=torch.long),
                seq_ids=seq_ids.to(device, dtype=torch.long),
                category_ids=category_ids.to(device, dtype=torch.long),
                note_end_chunk_ids=note_end_chunk_ids,
            )

            ids.append(data["hadm_id"][0].item())
            avail_doc_count.append(avail_docs)
            hyps.append(scores[-1, :].detach().cpu().numpy())
            refs.append(labels.detach().cpu().numpy())

        if optimise_threshold:
            pred_cutoff = mymetrics.get_optimal_microf1_threshold_v2(
                np.asarray(hyps), np.asarray(refs)
            )

        computed_results = mymetrics.from_numpy(
            np.asarray(hyps), np.asarray(refs), pred_cutoff=pred_cutoff
        )
  
    return computed_results
