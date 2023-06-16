import json
from data.preprocess import DataProcessor
import os 
from data.custom_dataset import CustomDataset
from data.utils import get_dataset, get_tokenizer
def others():
  tokenizer = get_tokenizer(config['base_checkpoint'])

  dataset_config = {
      "max_chunks" : config["max_chunks"],
      "priority_mode" : config["priority_mode"],
      "priority_idxs" : config["priority_idxs"]
  }

  training_set = get_dataset(notes_agg_df[:1000], "TRAIN", tokenizer = tokenizer, **dataset_config)
  training_generator = get_dataloader(training_set)

  validation_set = get_dataset(notes_agg_df, "VALIDATION", tokenizer = tokenizer, **dataset_config)
  validation_generator = get_dataloader(validation_set)

  test_set = get_dataset(notes_agg_df, "TEST", tokenizer = tokenizer, **dataset_config)
  test_generator = get_dataloader(test_set)

  ########### Run only if CPU!!!!!!!!!!!

  import os
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  ###################################


  model = ModelV2(model_name = config['base_checkpoint'],
                  num_labels = config['num_labels'],
                  num_attention_heads = config['attention_heads'],
                  num_layers = config['num_layers'],
                  max_chunks = config['max_chunks'],
                  use_positional_embeddings = config['use_positional_embeddings'],
                  use_reverse_positional_embeddings = config['use_reverse_positional_embeddings'],
                  use_document_embeddings = config['use_document_embeddings'],
                  use_reverse_document_embeddings = config['use_reverse_document_embeddings'],
                  use_category_embeddings = config['use_category_embeddings'],
                  use_all_tokens = config['use_all_tokens']
  )


  optimizer = optim.AdamW(model.parameters(), lr = config['lr'],
                                weight_decay = 0
                                )

  steps_per_epoch = int(np.ceil(len(training_generator) // config['grad_accumulation']))
  #steps_per_epoch = 1
  lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = config['lr'], three_phase = True,
                                                  total_steps = config['max_epochs'] * steps_per_epoch)

  scaler = GradScaler()


  ########### MY CODE: LOAD FROM CHECKPOINT AND CONTINUE TRAINING ################

  TOTAL_COMPLETED_EPOCHS = 0
  CURRENT_BEST = 0
  CURRENT_PATIENCE_COUNT = 0

  if config['load_from_checkpoint']:
    checkpoint = torch.load(os.path.join(project_path,f"results/{config['checkpoint_name']}.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Move optimizer to GPU
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.cuda()

    TOTAL_COMPLETED_EPOCHS = checkpoint['epochs']
    CURRENT_BEST = checkpoint['current_best']
    CURRENT_PATIENCE_COUNT = checkpoint['current_patience_count']

  else:
    pd.DataFrame({}).to_csv(os.path.join(project_path,f"results/{RUN_NAME}.csv")) # Create dummy csv because of GDrive bug

  results = {}


  #################################################################################

  train_part(model,
              optimizer,
              scaler,
              lr_scheduler,
            config,
              TOTAL_COMPLETED_EPOCHS,
              CURRENT_BEST,
              CURRENT_PATIENCE_COUNT,
              grad_accumulation_steps = config['grad_accumulation'],
              epochs = config['max_epochs'])


if __name__ == "__main__":
  RUN_NAME = "Run_all_notes_last_second_transf"
  config = {}

  config['base_checkpoint'] = os.path.join('', "RoBERTa-base-PM-M3-Voc-hf")
  config['attention_heads'] = 1
  config['num_layers'] = 1

  config['lr'] = 5e-5
  config['max_chunks'] = 2
  config['grad_accumulation'] = 2
  config['use_positional_embeddings'] = True
  config['use_reverse_positional_embeddings'] = True
  config['priority_mode'] = "Last"
  config["priority_idxs"] = [1]
  config['use_document_embeddings'] = True
  config['use_reverse_document_embeddings'] = True
  config['use_category_embeddings'] = True
  config['num_labels'] = 50
  config['use_all_tokens'] = True
  config['final_aggregation'] = 'cls'
  config['only_discharge_summary'] = False


  config['patience_threshold'] = 3
  config['max_epochs'] = 20
  config['save_model'] = False

  config['load_from_checkpoint'] = False
  config['checkpoint_name'] ="Run_all_notes_last_second_transf"

  with open(os.path.join('',f"results/config_{RUN_NAME}.json"),'w') as f:
    json.dump(config, f)
  
  dp = DataProcessor(dataset_path = 'dataset', config=config)
  notes_agg_df = dp.aggregate_data()
  tokenizer = get_tokenizer(config['base_checkpoint'])
  dataset = get_dataset(
    notes_agg_df,
    split='TRAIN',
    tokenizer=tokenizer,
    max_chunks = config['max_chunks']
  )

  item = dataset.__getitem__(9)
