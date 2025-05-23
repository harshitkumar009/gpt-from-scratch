import json
import time
import torch
import pathlib
import tiktoken
from gpt.build_llm.gpt2 import GPTModel
from gpt.build_llm.utils import train_model
from gpt.build_llm.data_preprocessing import create_dataloader

current_path = pathlib.Path(__file__).resolve().parent.parent.parent
model_type = "gpt2"

tokenizer = tiktoken.get_encoding(model_type)

with open('../../base_config.json', 'r') as f:
    configs = json.load(f)
    GPT_CONFIG = configs["base_configs"]
    GPT_CONFIG.update(configs["model_configs"][model_type])

"""
Loading the raw text file used for training LLM
"""
file = open(current_path/"datasets/harry_potter.txt","r",encoding = "utf-8")
raw_text = file.read()

"""
training and validation data as a dataloader object
"""
text_data = raw_text
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

"""
Training build_llm from scratch with the below hyperparameters
"""
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")