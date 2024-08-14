# tests/test_llama_equivalence.py

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, LlamaForCausalLM as HFLlamaForCausalLM
import torch

from port.l3_eqx import (
    LlamaEmbedding, LlamaLinear, LlamaRotaryEmbedding, LlamaRMSNorm,
    LlamaSdpaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel,
    LlamaForCausalLM, LlamaConfig
)

# Helper function to convert PyTorch tensor to JAX array
def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())

# Helper function to compare PyTorch and JAX outputs
def assert_close(torch_output, jax_output, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(torch_output.detach().numpy(), jax_output, rtol=rtol, atol=atol)

@pytest.fixture(scope="module")
def hf_model():
    model_name = "meta-llama/Llama-3-8b-hf"  # You might need to adjust this based on your access and requirements
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
    )
    return config

def test_llama_embedding(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_embed = hf_model.model.embed_tokens
    eqx_embed = LlamaEmbedding(eqx_config.vocab_size, eqx_config.hidden_size)
    
    # Copy weights
    eqx_embed.weight = torch_to_jax(hf_embed.weight)
    
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    hf_output = hf_embed(torch.tensor(input_ids.tolist()))
    eqx_output = eqx_embed(input_ids)
    
    assert_close(hf_output, eqx_output)

def test_llama_linear(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_linear = hf_model.model.layers[0].self_attn.q_proj
    eqx_linear = LlamaLinear(eqx_config.hidden_size, eqx_config.hidden_size, bias=False)
    
    # Copy weights
    eqx_linear.weight = torch_to_jax(hf_linear.weight)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, eqx_config.hidden_size))
    hf_output = hf_linear(torch.tensor(x.tolist()))
    eqx_output = eqx_linear(x)
    
    assert_close(hf_output, eqx_output)

def test_llama_rms_norm(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_norm = hf_model.model.norm
    eqx_norm = LlamaRMSNorm(eqx_config.hidden_size, eqx_config.rms_norm_eps)
    
    # Copy weights
    eqx_norm.weight = torch_to_jax(hf_norm.weight)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, eqx_config.hidden_size))
    hf_output = hf_norm(torch.tensor(x.tolist()))
    eqx_output = eqx_norm(x)
    
    assert_close(hf_output, eqx_output)

def test_llama_mlp(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_mlp = hf_model.model.layers[0].mlp
    eqx_mlp = LlamaMLP(eqx_config.hidden_size, eqx_config.intermediate_size)
    
    # Copy weights
    eqx_mlp.gate_proj.weight = torch_to_jax(hf_mlp.gate_proj.weight)
    eqx_mlp.up_proj.weight = torch_to_jax(hf_mlp.up_proj.weight)
    eqx_mlp.down_proj.weight = torch_to_jax(hf_mlp.down_proj.weight)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, eqx_config.hidden_size))
    hf_output = hf_mlp(torch.tensor(x.tolist()))
    eqx_output = eqx_mlp(x)
    
    assert_close(hf_output, eqx_output)

def test_llama_attention(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_attn = hf_model.model.layers[0].self_attn
    eqx_attn = LlamaSdpaAttention(
        eqx_config.hidden_size,
        eqx_config.num_attention_heads,
        eqx_config.num_key_value_heads,
        eqx_config.max_position_embeddings
    )
    
    # Copy weights
    eqx_attn.q_proj.weight = torch_to_jax(hf_attn.q_proj.weight)
    eqx_attn.k_proj.weight = torch_to_jax(hf_attn.k_proj.weight)
    eqx_attn.v_proj.weight = torch_to_jax(hf_attn.v_proj.weight)
    eqx_attn.o_proj.weight = torch_to_jax(hf_attn.o_proj.weight)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, eqx_config.hidden_size))
    position_ids = jnp.arange(5)[None, :]
    
    hf_output = hf_attn(torch.tensor(x.tolist()), position_ids=torch.tensor(position_ids.tolist()))[0]
    eqx_output = eqx_attn(x, position_ids=position_ids)
    
    assert_close(hf_output, eqx_output)

def test_llama_decoder_layer(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_layer = hf_model.model.layers[0]
    eqx_layer = LlamaDecoderLayer(eqx_config)
    
    # Copy weights
    eqx_layer.self_attn.q_proj.weight = torch_to_jax(hf_layer.self_attn.q_proj.weight)
    eqx_layer.self_attn.k_proj.weight = torch_to_jax(hf_layer.self_attn.k_proj.weight)
    eqx_layer.self_attn.v_proj.weight = torch_to_jax(hf_layer.self_attn.v_proj.weight)
    eqx_layer.self_attn.o_proj.weight = torch_to_jax(hf_layer.self_attn.o_proj.weight)
    eqx_layer.mlp.gate_proj.weight = torch_to_jax(hf_layer.mlp.gate_proj.weight)
    eqx_layer.mlp.up_proj.weight = torch_to_jax(hf_layer.mlp.up_proj.weight)
    eqx_layer.mlp.down_proj.weight = torch_to_jax(hf_layer.mlp.down_proj.weight)
    eqx_layer.input_layernorm.weight = torch_to_jax(hf_layer.input_layernorm.weight)
    eqx_layer.post_attention_layernorm.weight = torch_to_jax(hf_layer.post_attention_layernorm.weight)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, eqx_config.hidden_size))
    position_ids = jnp.arange(5)[None, :]
    
    hf_output = hf_layer(torch.tensor(x.tolist()), position_ids=torch.tensor(position_ids.tolist()))[0]
    eqx_output = eqx_layer(x, position_ids=position_ids)
    
    assert_close(hf_output, eqx_output)

def test_llama_model(hf_model, eqx_config):
    tokenizer, hf_model = hf_model
    eqx_model = LlamaModel(eqx_config)
    
    # Copy weights (this is a simplified version, you might need to implement a more thorough weight copying function)
    eqx_model.embed_tokens.weight = torch_to_jax(hf_model.model.embed_tokens.weight)
    eqx_model.norm.weight = torch_to_jax(hf_model.model.norm.weight)
    for i, layer in enumerate(eqx_model.layers):
        hf_layer = hf_model.model.layers[i]
        layer.self_attn.q_proj.weight = torch_to_jax(hf_layer.self_attn.q_proj.weight)
        layer.self_attn.k_proj.weight = torch_to_jax(hf_layer.self_attn.k_proj.weight)
        layer.self_attn.v_proj.weight = torch_to_jax(hf_layer.self_attn.v_proj.weight)
        layer.self_attn.o_proj.weight = torch_to_jax(hf_layer.self_attn.o_proj.weight)
        layer.mlp.gate_proj.weight = torch_to_jax(hf_layer.mlp.gate_proj.weight)
        layer.mlp.up_proj.weight = torch_to_jax(hf_layer.mlp.up_proj.weight)
        layer.mlp.down_proj.weight = torch_to_jax(hf_layer.mlp.down_proj.weight)
        layer.input_layernorm.weight = torch_to_jax(hf_layer.input_layernorm.weight)
        layer.post_attention_layernorm.weight = torch_to_jax(hf_layer.post_attention_layernorm.weight)
    
    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]
    
    hf_output = hf_model.model(input_ids, position_ids=position_ids)[0]
    eqx_output = eqx_model(jnp.array(input_ids), position_ids=jnp.array(position_ids))
    
    assert_close(hf_output, eqx_output)

def test_llama_for_causal_lm(hf_model, eqx_config):
    tokenizer, hf_model = hf_model
    eqx_model = LlamaForCausalLM(eqx_config)
    
    # Copy weights (this is a simplified version, you might need to implement a more thorough weight copying function)
    eqx_model.model.embed_tokens.weight = torch_to_jax(hf_model.model.embed_tokens.weight)
    eqx_model.model.norm.weight = torch_to_jax(hf_model.model.norm.weight)
    eqx_model.lm_head.weight = torch_to_jax(hf_model.lm_head.weight)
    for i, layer in enumerate(eqx_model.model.layers):
        hf_layer = hf_model.model.layers[i]
        layer.self_attn.q_proj.weight = torch_to_jax(hf_layer.self_attn.q_proj.weight)
        layer.self_attn.k_proj.weight = torch_to_jax(hf_layer.self_attn.k_proj.weight)
        layer.self_attn.v_proj.weight = torch_to_jax(hf_layer.self_attn.v_proj.weight)
        layer.self_attn.o_proj.weight = torch_to_jax(hf_layer.self_attn.o_proj.weight)
        layer.mlp.gate_proj.weight = torch_to_jax(hf_layer.mlp.gate_proj.weight)
        layer.mlp.up_proj.weight = torch_to_jax(hf_layer.mlp.up_proj.weight)
        layer.mlp.down_proj.weight = torch_to_jax(hf_layer.mlp.down_proj.weight)
        layer.input_layernorm.weight = torch_to_jax(hf_layer.input_layernorm.weight)
        layer.post_attention_layernorm.weight = torch_to_jax(hf_layer.post_attention_layernorm.weight)
    
    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]
    
    hf_output = hf_model(input_ids, position_ids=position_ids).logits
    eqx_output = eqx_model(jnp.array(input_ids), position_ids=jnp.array(position_ids))
    
    assert_close(hf_output, eqx_output)
