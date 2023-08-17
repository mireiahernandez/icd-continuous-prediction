import json
from data.preprocess import DataProcessor
import os
from data.custom_dataset import CustomDataset
from data.utils import get_dataset, get_tokenizer, get_dataloader
from model.model import Model
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import ast
import os
import itertools
import torch.utils.checkpoint
from torch.cuda.amp import GradScaler, autocast
from training.trainer import Trainer
import argparse
from evaluation.evaluate import evaluate
from evaluation.metrics import MyMetrics


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(
        description="Train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO: update the help definitions
    parser.add_argument("-r", "--run_name", type=str, default="test", help="run name")
  

    args = parser.parse_args()
    args_config = vars(args)

    # device
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    cpu = torch.device("cpu")
    print(device)

    ### for debugging use cpu
    # device = cpu

    config = {
        #    "run_name": "Run_test_TLWAN"
        "run_name": args_config["run_name"],
        "project_path": "/vol/bitbucket/mh1022/temporal-modelling-icd",
    }
    with open(os.path.join("", f"results/config_{config['run_name']}.json"), "r") as f:
        config = json.load(f)

    # process and aggregate raw data
    dp = DataProcessor(dataset_path="dataset", config=config)
    notes_agg_df, categories_mapping = dp.aggregate_data()

    # get tokenizer
    tokenizer = get_tokenizer(config["base_checkpoint"])

    # Get training / validation / test
    dataset_config = {
        "max_chunks": config["max_chunks"],
        "setup": config["setup"],
        "limit_ds": config["limit_ds"],
    }

    test_set = get_dataset(notes_agg_df, "TEST", tokenizer=tokenizer, **dataset_config)
    test_generator = get_dataloader(test_set)

    # only to run on CPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config["num_categories"] = len(categories_mapping)
    model = Model(config, device=device)


    # load best model
    checkpoint = torch.load(
        os.path.join(
            config["project_path"], f"results/BEST_{config['run_name']}.pth"
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # create TEST results csv
    cutoff_times = ["all", "2d", "5d", "13d", "noDS", "all_aux"]
    for time in cutoff_times:
        pd.DataFrame({}).to_csv(
            os.path.join(
                config["project_path"], f"results/TEST_{config['run_name']}_{time}.csv"
            )
        )  # Create dummy csv because of GDrive bug
    results = {}

    # evaluation metrics
    mymetrics = MyMetrics(debug=config["debug"])

    # evaluate on test set
    test_metrics, test_metrics_temp, test_metrics_aux = evaluate(
        mymetrics,
        model,
        test_generator,
        device,
        evaluate_temporal=config["evaluate_temporal"],
        optimise_threshold=False,
        num_categories=config["num_categories"],
        is_baseline=config["is_baseline"],
        aux_task=config["aux_task"],
        setup=config["setup"],
        reduce_computation=config["reduce_computation"],
    )
    breakpoint()
    test_metrics["f1_by_class"] = list(test_metrics["f1_by_class"])
    test_metrics["auc_by_class"] = list(test_metrics["auc_by_class"])
    # save all results
    with open(f"results/TEST_{config['run_name']}_all.csv", "w") as f:
        json.dump(test_metrics, f)

    
    # save temp results
    if config["evaluate_temporal"]:
        for time in ["2d", "5d", "13d", "noDS"]:
            test_metrics_temp[time]["f1_by_class"] = list(test_metrics_temp[time]["f1_by_class"])
            test_metrics_temp[time]["auc_by_class"] = list(test_metrics_temp[time]["auc_by_class"])
            with open(f"results/TEST_{config['run_name']}_{time}.csv", "w") as f:
                json.dump(test_metrics_temp[time], f)
    
    print(f"results: {test_metrics_temp}")