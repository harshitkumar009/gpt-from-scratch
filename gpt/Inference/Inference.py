import json
import torch
import tiktoken
from gpt.build_llm.gpt2 import GPTModel
from transformers import GPT2LMHeadModel
from gpt.model_weights.load_model_weights import load_weights
from generate_logic import generate_and_print_sample

model_type = "gpt2"
with open('../../base_config.json', 'r') as f:
    configs = json.load(f)
    GPT_CONFIG = configs["base_configs"].update(configs["model_configs"][model_type])

tokenizer = tiktoken.get_encoding(model_type)
model = GPTModel(GPT_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Loading foundational model weights using Hugging Face transformers library
"""
foundational_model = GPT2LMHeadModel.from_pretrained("gpt2")
state_dict = foundational_model.state_dict()
model = load_weights(n_layers=GPT_CONFIG["n_layers"], model=model, state_dict=state_dict)
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