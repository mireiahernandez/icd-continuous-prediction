from tqdm import tqdm
import torch
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F


def evaluate(
    mymetrics,
    model,
    generator,
    device,
    pred_cutoff=0.5,
    evaluate_temporal=False,
    optimise_threshold=False,
    diag_indices=None,
    proc_indices=None,
):
    model.eval()
    with torch.no_grad():
        ids = []
        hyps = []
        refs = []
        hyps_proc, hyps_diag, refs_proc, refs_diag = [], [], [], []

        if evaluate_temporal:
            hyps_temp ={'2d':[],'6d':[],'14d':[],'noDS':[]}
            refs_temp ={'2d':[],'6d':[],'14d':[],'noDS':[]}
          
        avail_doc_count = []
        print(f"Starting validation loop...")
        for t, data in enumerate(tqdm(generator)):
            if len(data["seq_ids"][0]) == 0:
                continue
            labels = data["label"][0][: model.num_labels]

            input_ids = data["input_ids"][0]
            attention_mask = data["attention_mask"][0]
            seq_ids = data["seq_ids"][0]
            category_ids = data["category_ids"][0]
            avail_docs = seq_ids.max().item() + 1
            # note_end_chunk_ids = data["note_end_chunk_ids"]
            cutoffs = data["cutoffs"]

            scores, _ = model(
                input_ids=input_ids.to(device, dtype=torch.long),
                attention_mask=attention_mask.to(device, dtype=torch.long),
                seq_ids=seq_ids.to(device, dtype=torch.long),
                category_ids=category_ids.to(device, dtype=torch.long),
                # note_end_chunk_ids=note_end_chunk_ids,
            )
            probs = F.sigmoid(scores)
            ids.append(data["hadm_id"][0].item())
            avail_doc_count.append(avail_docs)
            hyps.append(probs[-1, :].detach().cpu().numpy())
            refs.append(labels.detach().cpu().numpy())
            hyps_proc.append(probs[-1, :].detach().cpu().numpy()[proc_indices])
            refs_proc.append(labels.detach().cpu().numpy()[proc_indices])
            hyps_diag.append(probs[-1, :].detach().cpu().numpy()[diag_indices])
            refs_diag.append(labels.detach().cpu().numpy()[diag_indices])

            if evaluate_temporal:
                cutoff_times = ['2d','6d','14d','noDS']
                for time in cutoff_times:
                    if cutoffs[time] != -1:
                        hyps_temp[time].append(probs[cutoffs[time][0], :].detach().cpu().numpy())
                        refs_temp[time].append(labels.detach().cpu().numpy())

        if optimise_threshold:
            pred_cutoff = mymetrics.get_optimal_microf1_threshold_v2(
                np.asarray(hyps), np.asarray(refs)
            )

        val_metrics = mymetrics.from_numpy(np.asarray(hyps), np.asarray(refs), pred_cutoff=pred_cutoff)
        val_metrics_proc = mymetrics.from_numpy(np.asarray(hyps_proc), np.asarray(refs_proc), pred_cutoff=pred_cutoff)
        val_metrics_diag = mymetrics.from_numpy(np.asarray(hyps_diag), np.asarray(refs_diag), pred_cutoff=pred_cutoff)
        if evaluate_temporal:
            cutoff_times = ['2d', '6d', '14d', 'noDS']
            val_metrics_temp = {time: mymetrics.from_numpy(np.asarray(hyps_temp[time]), np.asarray(refs_temp[time])) for time in cutoff_times}
        else:
            val_metrics_temp = None
  
    return val_metrics, val_metrics_proc, val_metrics_diag, val_metrics_temp
