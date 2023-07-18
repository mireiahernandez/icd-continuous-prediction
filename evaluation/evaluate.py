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
    num_categories=1,
    is_baseline=False,
    aux_task=None,
    setup='latest',
):
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
                    cutoffs = cutoffs,
                    # note_end_chunk_ids=note_end_chunk_ids,
                )
            if aux_task == "next_document_category":
                if len(category_ids) > 1 and aux_predictions is not None:
                    true_categories = F.one_hot(
                        torch.concat([category_ids[1:], torch.tensor([num_categories])]),
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
                        preds["hyps_temp"][time].append(
                            probs[n, :].detach().cpu().numpy()
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

    return val_metrics, val_metrics_temp, val_metrics_aux
