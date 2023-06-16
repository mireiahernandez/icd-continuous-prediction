from data.custom_dataset import CustomDataset
import torch
import transformers

dataloader_params = {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': 6,
    'pin_memory': True
}

def get_dataset(df,
                split,
               tokenizer,
               max_chunks,
               priority_mode = "Last",
               priority_idxs = None):
  return CustomDataset(df[df.SPLIT == split], tokenizer = tokenizer, max_chunks = max_chunks, priority_mode = priority_mode, priority_idxs = priority_idxs)

def get_dataloader(dataset):
  return torch.utils.data.DataLoader(dataset, **dataloader_params)

def get_tokenizer(checkpoint):
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint,
                                                         model_max_length = 512)
  return tokenizer


