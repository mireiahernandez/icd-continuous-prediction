import tqdm
import torch
import pandas as pd
import os
import numpy as np


def evaluate(
    mymetrics,
    model,
    RUN_NAME,
    TOTAL_COMPLETED_EPOCHS,
    project_path,
    generator,
    device,
    pred_cutoff=0.5,
    optimise_threshold=False,
):
    try:
        scores_df = pd.read_csv(f"outputs/scores_{RUN_NAME}.csv")
    except:
        scores_df = pd.DataFrame({})

    model.eval()
    with torch.no_grad():
        ids = []
        hyps = []
        refs = []
        avail_doc_count = []

        for t, data in enumerate(tqdm(generator)):
            labels = data["label"][0][: model.num_labels]

            input_ids = data["input_ids"][0]
            attention_mask = data["attention_mask"][0]
            seq_ids = data["seq_ids"][0]
            category_ids = data["category_ids"][0]
            avail_docs = seq_ids.max().item() + 1

            scores = model(
                input_ids=input_ids.to(device, dtype=torch.long),
                attention_mask=attention_mask.to(device, dtype=torch.long),
                seq_ids=seq_ids.to(device, dtype=torch.long),
                category_ids=category_ids.to(device, dtype=torch.long),
            )

            ids.append(data["hadm_id"][0].item())
            avail_doc_count.append(avail_docs)
            hyps.append(scores.detach().cpu().numpy())
            refs.append(labels.detach().cpu().numpy())

        if optimise_threshold:
            pred_cutoff = mymetrics.get_optimal_microf1_threshold_v2(
                np.asarray(hyps), np.asarray(refs)
            )

        computed_results = mymetrics.from_numpy(
            np.asarray(hyps), np.asarray(refs), pred_cutoff=pred_cutoff
        )
        scores_df_tmp = pd.DataFrame(
            {"id": ids, "pred_scores": hyps, "avail_doc_count": avail_doc_count}
        )
        scores_df_tmp["epochs"] = TOTAL_COMPLETED_EPOCHS
        scores_df = pd.concat([scores_df, scores_df_tmp])
        scores_df.to_csv(
            os.path.join(project_path, f"outputs/scores_{RUN_NAME}.csv"), index=False
        )

    return computed_results
