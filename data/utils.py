from data.custom_dataset import CustomDataset
import torch
import transformers


def get_dataset(df, split, tokenizer, max_chunks, setup, limit_ds=0):
    return CustomDataset(
        df[df.SPLIT == split],
        tokenizer=tokenizer,
        max_chunks=max_chunks,
        setup=setup,
        limit_ds=limit_ds,
    )


def get_dataloader(dataset):
    dataloader_params = {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 6,
        "pin_memory": True,
    }
    return torch.utils.data.DataLoader(dataset, **dataloader_params)


def get_tokenizer(checkpoint):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, model_max_length=512
    )
    return tokenizer
