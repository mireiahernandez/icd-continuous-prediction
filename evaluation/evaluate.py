from tqdm import tqdm
import torch
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F
import json
import ipdb


def return_attn_scores(lwan, encoding, all_tokens=True, cutoffs=None):
    # encoding: Tensor of size (Nc x T) x H
    # mask: Tensor of size Nn x (Nc x T) x H
    # temporal_encoding = Nn x (N x T) x hidden_size
    T = lwan.seq_len
    if not lwan.all_tokens:
        T = 1  # only use the [CLS]-token representation
    Nc = int(encoding.shape[0] / T)
    H = lwan.hidden_size
    Nl = lwan.num_labels

    # label query: shape L, H
    # encoding: hape NcxT, H
    # query shape:  Nn, L, H
    # key shape: Nn, Nc*T, H
    # values shape: Nn, Nc*T, H
    # key padding mask: Nn, Nc*T (true if ignore)
    # output: N, L, H
    mask = torch.ones(size=(Nc, Nc * T), dtype=torch.bool).to(device=lwan.device)
    for i in range(Nc):
        mask[i, : (i + 1) * T] = False

    # only mask out at 2d, 5d, 13d and no DS to reduce computation
    # get list of cutoff indices from cutoffs dictionary

    attn_output, attn_output_weights = lwan.multiheadattn.forward(
        query=lwan.label_queries.repeat(mask.shape[0], 1, 1),
        key=encoding.repeat(mask.shape[0], 1, 1),
        value=encoding.repeat(mask.shape[0], 1, 1),
        key_padding_mask=mask,
        need_weights=True,
    )

    score = torch.sum(
        attn_output
        * lwan.label_weights.unsqueeze(0).view(
            1, lwan.num_labels, lwan.hidden_size
        ),
        dim=2,
    )
    return attn_output_weights, score


def update_weights_per_class(labels, cutoffs, category_ids, attn_output_weights, weights_per_class):
    labels_sample = []
    for i in range(50):
        if labels[i] == 1:
            labels_sample.append(i)
    for cutoff in cutoffs.keys():
        cutoff_idx = cutoffs[cutoff]   
        for l in labels_sample:   
            attn_weights= attn_output_weights[cutoff_idx, l, :].cpu().detach().numpy().reshape(1, -1)
            for chunk in range(cutoff_idx+1):
                c = category_ids[chunk].item()
                weights_per_class[cutoff][c].append( attn_output_weights[cutoff_idx, l, chunk].item())   
    # update the 'all' key
    cutoff_idx = attn_output_weights.shape[0]-1
    for l in labels_sample:   
        attn_weights= attn_output_weights[cutoff_idx, l, :].cpu().detach().numpy().reshape(1, -1)
        for chunk in range(cutoff_idx+1):
            c = category_ids[chunk].item()
            weights_per_class['all'][c].append( attn_output_weights[cutoff_idx, l, chunk].item())
    return weights_per_class

def evaluate(
    mymetrics,
    model,
    generator,
    device,
    pred_cutoff=0.5,
    evaluate_temporal=False,
    optimise_threshold=False,
    num_categories=1,
    is_baseline=False,
    aux_task=None,
    setup="latest",
    reduce_computation=False,
    qualitative_evaluation=False,
):
    if qualitative_evaluation:
        weights_per_class = {cutoff: {c: [] for c in range(15)} for cutoff in ["2d", "5d", "13d", "noDS", 'all']}

    model.eval()
    with torch.no_grad():
        ids = []
        preds = {"hyps": [], "refs": [], "hyps_aux": [], "refs_aux": []}

        if evaluate_temporal:
            preds["hyps_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}
            preds["refs_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}

        avail_doc_count = []
        print(f"Starting validation loop...")
        for t, data in enumerate(tqdm(generator)):
            # TODO: fix code so that sequence ids embeddings can be used
            # right now they cannot be used
            if setup == "random":
                labels = data["label"][0][: model.num_labels]
                input_ids = data["input_ids"][0]
                attention_mask = data["attention_mask"][0]
                seq_ids = data["seq_ids"][0]
                category_ids = data["category_ids"][0]
                avail_docs = seq_ids.max().item() + 1
                # note_end_chunk_ids = data["note_end_chunk_ids"]
                cutoffs = data["cutoffs"]

                complete_sequence_output = []
                # run through data in chunks of max_chunks
                for i in range(0, input_ids.shape[0], model.max_chunks):
                    # only get the document embeddings
                    sequence_output = model(
                        input_ids=input_ids[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        attention_mask=attention_mask[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        seq_ids=seq_ids[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        category_ids=category_ids[i : i + model.max_chunks].to(
                            device, dtype=torch.long
                        ),
                        cutoffs=None,
                        is_evaluation=True,
                        # note_end_chunk_ids=note_end_chunk_ids,
                    )
                    complete_sequence_output.append(sequence_output)
                # concatenate the sequence output
                sequence_output = torch.cat(complete_sequence_output, dim=0)

                # run through LWAN to get the scores
                scores = model.label_attn(sequence_output, cutoffs=cutoffs)
                if qualitative_evaluation:
                    attn_output_weights, scores = return_attn_scores(model.label_attn, sequence_output.to(device), cutoffs=cutoffs)
                    weights_per_class = update_weights_per_class(labels, cutoffs, category_ids, attn_output_weights, weights_per_class)

            else:
                labels = data["label"][0][: model.num_labels]
                input_ids = data["input_ids"][0]
                attention_mask = data["attention_mask"][0]
                seq_ids = data["seq_ids"][0]
                category_ids = data["category_ids"][0]
                avail_docs = seq_ids.max().item() + 1
                # note_end_chunk_ids = data["note_end_chunk_ids"]
                cutoffs = data["cutoffs"]

                scores, _, aux_predictions = model(
                    input_ids=input_ids.to(device, dtype=torch.long),
                    attention_mask=attention_mask.to(device, dtype=torch.long),
                    seq_ids=seq_ids.to(device, dtype=torch.long),
                    category_ids=category_ids.to(device, dtype=torch.long),
                    cutoffs=cutoffs,
                    # note_end_chunk_ids=note_end_chunk_ids,
                )
            if aux_task == "next_document_category":
                if len(category_ids) > 1 and aux_predictions is not None:
                    true_categories = F.one_hot(
                        torch.concat(
                            [category_ids[1:], torch.tensor([num_categories])]
                        ),
                        num_classes=num_categories + 1,
                    )
                    preds["hyps_aux"].append(aux_predictions.detach().cpu().numpy())
                    preds["refs_aux"].append(true_categories.detach().cpu().numpy())

            probs = F.sigmoid(scores)
            ids.append(data["hadm_id"][0].item())
            avail_doc_count.append(avail_docs)
            preds["hyps"].append(probs[-1, :].detach().cpu().numpy())
            preds["refs"].append(labels.detach().cpu().numpy())
            if evaluate_temporal:
                cutoff_times = ["2d", "5d", "13d", "noDS"]
                for n, time in enumerate(cutoff_times):
                    if cutoffs[time][0] != -1:
                        if reduce_computation:
                            preds["hyps_temp"][time].append(
                                probs[n, :].detach().cpu().numpy()
                            )
                        else:
                            preds["hyps_temp"][time].append(
                                probs[cutoffs[time][0], :].detach().cpu().numpy()
                            )
                        preds["refs_temp"][time].append(labels.detach().cpu().numpy())

        if optimise_threshold:
            pred_cutoff = mymetrics.get_optimal_microf1_threshold_v2(
                np.asarray(preds["hyps"]), np.asarray(preds["refs"])
            )
        else:
            pred_cutoff = 0.5

        val_metrics = mymetrics.from_numpy(
            np.asarray(preds["hyps"]),
            np.asarray(preds["refs"]),
            pred_cutoff=pred_cutoff,
        )
        if len(preds["hyps_aux"]) > 0:
            val_metrics_aux = mymetrics.from_numpy(
                np.concatenate(preds["hyps_aux"]),
                np.concatenate(preds["refs_aux"]),
                pred_cutoff=pred_cutoff,
            )
        else:
            val_metrics_aux = None

        if evaluate_temporal:
            cutoff_times = ["2d", "5d", "13d", "noDS"]
            val_metrics_temp = {
                time: mymetrics.from_numpy(
                    np.asarray(preds["hyps_temp"][time]),
                    np.asarray(preds["refs_temp"][time]),
                    pred_cutoff=pred_cutoff,
                )
                for time in cutoff_times
            }
        else:
            val_metrics_temp = None
        
        if qualitative_evaluation:
            json.dump(weights_per_class, open("weights_per_class_3.json",'w'))    


    return val_metrics, val_metrics_temp, val_metrics_aux
