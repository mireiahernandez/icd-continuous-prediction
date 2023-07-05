from data.custom_dataset import CustomDataset
import torch
import transformers



def get_dataset(df,
                split,
               tokenizer,
               max_chunks,
               priority_mode = "Last",
               priority_idxs = None,
               filter_time = "all"):
  return CustomDataset(df[df.SPLIT == split], tokenizer = tokenizer, max_chunks = max_chunks, priority_mode = priority_mode, priority_idxs = priority_idxs, filter_time=filter_time)

def get_dataloader(dataset):
  dataloader_params = {
      'batch_size': 1,
      'shuffle': True,
      'num_workers': 6,
      'pin_memory': True
  }
  return torch.utils.data.DataLoader(dataset, **dataloader_params)

def get_tokenizer(checkpoint):
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint,
                                                         model_max_length = 512)
  return tokenizer


