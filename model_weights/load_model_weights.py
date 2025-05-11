import torch
import numpy as np


"""
Below is the implementation for loading already trained model weights to our GPT model
"""
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights(n_layers, model, state_dict):
    model.tok_emb.weight = assign(model.tok_emb.weight, (state_dict['transformer.wte.weight']))
    model.pos_emb.weight = assign(model.pos_emb.weight, (state_dict['transformer.wpe.weight']))
    for i in range(n_layers):
        model.trf_blocks[i].norm1.scale = assign(model.trf_blocks[i].norm1.scale, (state_dict[f'transformer.h.{i}.ln_1.weight']))
        model.trf_blocks[i].norm1.shift = assign(model.trf_blocks[i].norm1.shift, (state_dict[f'transformer.h.{i}.ln_1.bias']))

        q_w, k_w, v_w = np.split(state_dict[f'transformer.h.{i}.attn.c_attn.weight'],3,axis=-1)
        q_b, k_b, v_b = np.split(state_dict[f'transformer.h.{i}.attn.c_attn.bias'],3,axis=-1)
        model.trf_blocks[i].att.W_query.weight = assign(model.trf_blocks[i].att.W_query.weight, (q_w.T))
        model.trf_blocks[i].att.W_key.weight = assign(model.trf_blocks[i].att.W_key.weight, (k_w.T))
        model.trf_blocks[i].att.W_value.weight = assign(model.trf_blocks[i].att.W_value.weight, (v_w.T))
        model.trf_blocks[i].att.W_query.bias = assign(model.trf_blocks[i].att.W_query.bias, (q_b))
        model.trf_blocks[i].att.W_key.bias = assign(model.trf_blocks[i].att.W_key.bias, (k_b))
        model.trf_blocks[i].att.W_value.bias = assign(model.trf_blocks[i].att.W_value.bias, (v_b))

        model.trf_blocks[i].att.out_proj.weight = assign(model.trf_blocks[i].att.out_proj.weight, (state_dict[f'transformer.h.{i}.attn.c_proj.weight']).T)
        model.trf_blocks[i].att.out_proj.bias = assign(model.trf_blocks[i].att.out_proj.bias, (state_dict[f'transformer.h.{i}.attn.c_proj.bias']))

        model.trf_blocks[i].norm2.scale = assign(model.trf_blocks[i].norm2.scale, (state_dict[f'transformer.h.{i}.ln_2.weight']))
        model.trf_blocks[i].norm2.bias = assign(model.trf_blocks[i].norm2.scale, (state_dict[f'transformer.h.{i}.ln_2.bias']))

        model.trf_blocks[i].ff.layers[0].weight = assign(model.trf_blocks[i].ff.layers[0].weight, (state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].T))
        model.trf_blocks[i].ff.layers[0].bias = assign(model.trf_blocks[i].ff.layers[0].bias, (state_dict[f'transformer.h.{i}.mlp.c_fc.bias']))
        model.trf_blocks[i].ff.layers[2].weight = assign(model.trf_blocks[i].ff.layers[2].weight, (state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].T))
        model.trf_blocks[i].ff.layers[2].bias = assign(model.trf_blocks[i].ff.layers[2].bias, (state_dict[f'transformer.h.{i}.mlp.c_proj.bias']))

    model.final_norm.scale = assign(model.final_norm.scale, (state_dict['transformer.ln_f.weight']))
    model.final_norm.shift = assign(model.final_norm.shift, (state_dict['transformer.ln_f.bias']))
    model.out_head.weight = assign(model.out_head.weight, (state_dict['lm_head.weight']))
    return model