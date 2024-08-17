"""
test_l3_70b_equivalence.py

This script provides a simplified test suite for validating the equivalence
between the original PyTorch implementation of the Llama-3-70B model and its
JAX/Equinox port. It focuses on testing the full LlamaForCausalLM model.

Usage:
    Run this script using pytest to validate the equivalence of the PyTorch and
    JAX/Equinox implementations of the Llama-3-70B model.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, LlamaForCausalLM as HFLlamaForCausalLM
import torch
import equinox as eqx

from port.l3_eqx import LlamaForCausalLM, LlamaConfig

# Helper function to convert PyTorch tensor to JAX array
def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())

# Helper function to compare PyTorch and JAX outputs
def assert_close(torch_output, jax_output, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(torch_output.detach().numpy(), jax_output, rtol=rtol, atol=atol)

@pytest.fixture(scope="module")
def hf_model():
    model_name = "meta-llama/Meta-Llama-3-70B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model

@pytest.fixture(scope="module")
def eqx_config(hf_model):
    _, hf_model = hf_model
    config = LlamaConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        intermediate_size=hf_model.config.intermediate_size,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        num_attention_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        rms_norm_eps=hf_model.config.rms_norm_eps,
        rope_theta=hf_model.config.rope_theta,
        attention_bias=hf_model.config.attention_bias
    )
    return config

def test_llama_for_causal_lm(hf_model, eqx_config):
    tokenizer, hf_model = hf_model
    eqx_model = LlamaForCausalLM(eqx_config)
    
    # Copy weights
    eqx_model = eqx.tree_at(lambda t: t.model.embed_tokens.weight, eqx_model, torch_to_jax(hf_model.model.embed_tokens.weight))
    eqx_model = eqx.tree_at(lambda t: t.model.norm.weight, eqx_model, torch_to_jax(hf_model.model.norm.weight))
    eqx_model = eqx.tree_at(lambda t: t.lm_head.weight, eqx_model, torch_to_jax(hf_model.lm_head.weight))
    
    for i, layer in enumerate(eqx_model.model.layers):
        hf_layer = hf_model.model.layers[i]
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].self_attn.q_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.q_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].self_attn.k_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.k_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].self_attn.v_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.v_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].self_attn.o_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.o_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].mlp.gate_proj.weight, eqx_model, torch_to_jax(hf_layer.mlp.gate_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].mlp.up_proj.weight, eqx_model, torch_to_jax(hf_layer.mlp.up_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].mlp.down_proj.weight, eqx_model, torch_to_jax(hf_layer.mlp.down_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].input_layernorm.weight, eqx_model, torch_to_jax(hf_layer.input_layernorm.weight))
        eqx_model = eqx.tree_at(lambda t: t.model.layers[i].post_attention_layernorm.weight, eqx_model, torch_to_jax(hf_layer.post_attention_layernorm.weight))
    
    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]
    
    # PyTorch forward pass
    with torch.no_grad():
        hf_output = hf_model(input_ids, position_ids=position_ids).logits
    
    # JAX forward pass
    eqx_output = eqx_model(jnp.array(input_ids), position_ids=jnp.array(position_ids))
    
    # Compare outputs
    assert_close(hf_output, eqx_output, rtol=1e-4, atol=1e-4)
    
    print("PyTorch and JAX/Equinox outputs match within tolerance.")
