import json
import torch
import tiktoken
from GPT.GPT2 import GPTModel
from transformers import GPT2LMHeadModel
from model_weights.load_model_weights import load_weights
from generate_logic import generate_and_print_sample

with open('../base_config.json', 'r') as f:
    GPT_CONFIG_124M = json.load(f)

tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Loading foundational model weights using Hugging Face transformers library
"""
foundational_model = GPT2LMHeadModel.from_pretrained("gpt2")
state_dict = foundational_model.state_dict()
model = load_weights(n_layers=GPT_CONFIG_124M["n_layers"],model=model,state_dict=state_dict)
"""
Loading the locally trained weights trained on only harry_potter.txt dataset
"""
# model.load_state_dict(torch.load('../model_weights/model_weights.pth',map_location=device))

model.eval()
start_context = "hi how are you"
max_new_tokens = 15
temperature = 1.2
top_k = 25
generate_and_print_sample(model, tokenizer, device, start_context,max_new_tokens,temperature,top_k,eos_id=None)