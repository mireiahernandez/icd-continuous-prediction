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

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Train model",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_chunks", type=int, help="number of chunks")
    parser.add_argument("-r", "--run_name", type=str, help="run name")
    args = parser.parse_args()
    args_config = vars(args)

    # device
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    cpu = torch.device('cpu')
    print(device)

    ### for debugging use cpu
    #device = cpu

    config = {
    #    "run_name": "Run_test_TLWAN"
        "run_name": args_config['run_name']
        ,"project_path": '/vol/bitbucket/mh1022/temporal-modelling-icd'
        ,"base_checkpoint": os.path.join("", "RoBERTa-base-PM-M3-Voc-hf")
        ,"num_attention_heads": 1
        ,"num_layers": 1
        ,"lr": 5e-5
        ,"max_chunks": args_config['num_chunks']
        ,"grad_accumulation": args_config['num_chunks']
        ,"use_positional_embeddings": True
        ,"use_reverse_positional_embeddings": True
        ,"priority_mode": "Last"
        ,"priority_idxs": [1]
        ,"use_document_embeddings": True
        ,"use_reverse_document_embeddings": True
        ,"use_category_embeddings": True
        ,"num_labels": 50
        ,"use_all_tokens": True
        ,"final_aggregation": "cls"
        ,"only_discharge_summary": False
        ,"patience_threshold": 3
        ,"max_epochs": 3
        ,"save_model": False
        ,"load_from_checkpoint": False
        ,"checkpoint_name": "Run_all_notes_last_second_transf"
    }

    with open(os.path.join("", f"results/config_{config['run_name']}.json"), "w") as f:
        json.dump(config, f)

    # process and aggregate raw data
    dp = DataProcessor(dataset_path="dataset", config=config)
    notes_agg_df = dp.aggregate_data()
    
    # get tokenizer
    tokenizer = get_tokenizer(config["base_checkpoint"])

    # Get training / validation / test
    dataset_config = {
        "max_chunks" : config["max_chunks"],
        "priority_mode" : config["priority_mode"],
        "priority_idxs" : config["priority_idxs"]
    }
    training_set = get_dataset(notes_agg_df, "TRAIN", tokenizer = tokenizer, **dataset_config)
    training_generator = get_dataloader(training_set)
    
    validation_set =  get_dataset(notes_agg_df, "VALIDATION", tokenizer = tokenizer, **dataset_config)
    validation_generator = get_dataloader(validation_set)

    test_set = get_dataset(notes_agg_df, "TEST", tokenizer = tokenizer, **dataset_config)
    test_generator = get_dataloader(test_set)

    # validation_set = get_dataset(notes_agg_df, "VALIDATION", tokenizer = tokenizer, **dataset_config)
    # validation_generscator = get_dataloader(validation_set)

    # test_set = get_dataset(notes_agg_df, "TEST", tokenizer = tokenizer, **dataset_config)
    # test_generator = get_dataloader(test_set)

    # only to run on CPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = Model(config, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0)

    steps_per_epoch = int(
        np.ceil(len(training_generator) // config["grad_accumulation"])
    )

    # steps_per_epoch = 1
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        three_phase=True,
        total_steps=config["max_epochs"] * steps_per_epoch
    )

    scaler = GradScaler()

    ########### MY CODE: LOAD FROM CHECKPOINT AND CONTINUE TRAINING ################
    training_args = {
        'TOTAL_COMPLETED_EPOCHS': 0,
        'CURRENT_BEST': 0,
        'CURRENT_PATIENCE_COUNT': 0,
    }

    if config["load_from_checkpoint"]:
        checkpoint = torch.load(
            os.path.join(config['project_path'], f"results/{config['checkpoint_name']}.pth")
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Move optimizer to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        training_args['TOTAL_COMPLETED_EPOCHS'] = checkpoint["epochs"]
        training_args['CURRENT_BEST'] = checkpoint["current_best"]
        training_args['CURRENT_PATIENCE_COUNT'] = checkpoint["current_patience_count"]

    else:
        pd.DataFrame({}).to_csv(
            os.path.join(config['project_path'], f"results/{config['run_name']}.csv")
        )  # Create dummy csv because of GDrive bug

    results = {}

        #################################################################################
    trainer = Trainer(
        model,
        optimizer,
        scaler,
        lr_scheduler,
        config,
        device,
        dtype
    )
    trainer.train(
        training_generator,
        training_args,
        validation_generator,
        grad_accumulation_steps=config["grad_accumulation"],
        epochs=config["max_epochs"],
    )