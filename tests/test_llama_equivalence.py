"""
test_llama_equivalence.py

This script serves as a comprehensive test suite for validating the equivalence
between the original PyTorch implementation of the Llama model and its JAX/Equinox
port. It performs the following key tasks:

1. Imports both the Hugging Face Transformers Llama model and the custom JAX/Equinox
   implementation.
2. Defines test functions for each major component of the Llama architecture, including:
   - Token Embedding
   - Linear Transformations
   - RMS Normalization
   - Multi-Layer Perceptron (MLP)
   - Self-Attention Mechanism
   - Decoder Layer
   - Full Model
   - Causal Language Model

3. For each component, the test:
   - Initializes both PyTorch and JAX/Equinox versions
   - Copies weights from the PyTorch model to the JAX/Equinox model
   - Generates identical inputs for both versions
   - Computes outputs using both implementations
   - Asserts that the outputs are numerically close, within a specified tolerance

The primary purposes of this script are to:
- Ensure that the JAX/Equinox implementation correctly replicates the behavior of
  the original PyTorch model.
- Verify that each component of the Llama architecture has been accurately ported.
- Catch any discrepancies or errors in the porting process.
- Provide a reliable test suite for ongoing development and refactoring of the
  JAX/Equinox implementation.

Usage:
    Run this script using pytest to validate the equivalence of the PyTorch and
    JAX/Equinox implementations of the Llama model. All tests should pass if the
    porting process has been successful.

Note:
    This test suite is crucial for maintaining the integrity and accuracy of the
    JAX/Equinox port. It should be run after any significant changes to the
    implementation and as part of the continuous integration process.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, LlamaForCausalLM as HFLlamaForCausalLM
import torch
import equinox as eqx

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
    model_name = "meta-llama/Meta-Llama-3-8B"  # You might need to adjust this based on your access and requirements
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
    eqx_embed = eqx.tree_at(lambda t: t.weight, eqx_embed, torch_to_jax(hf_embed.weight))
    
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    hf_output = hf_embed(torch.tensor(input_ids.tolist()))
    eqx_output = eqx_embed(input_ids)
    
    assert_close(hf_output, eqx_output)

def test_llama_linear(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_linear = hf_model.model.layers[0].self_attn.q_proj
    eqx_linear = LlamaLinear(eqx_config.hidden_size, eqx_config.hidden_size, bias=False)
    
    # Copy weights
    eqx_linear = eqx.tree_at(lambda t: t.weight, eqx_linear, torch_to_jax(hf_linear.weight))
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, eqx_config.hidden_size))
    hf_output = hf_linear(torch.tensor(x.tolist()))
    eqx_output = eqx_linear(x)
    
    assert_close(hf_output, eqx_output)

def test_llama_rms_norm(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_norm = hf_model.model.norm
    eqx_norm = LlamaRMSNorm(eqx_config.hidden_size, eqx_config.rms_norm_eps)
    
    # Copy weights
    eqx_norm = eqx.tree_at(lambda t: t.weight, eqx_norm, torch_to_jax(hf_norm.weight))
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, eqx_config.hidden_size))
    hf_output = hf_norm(torch.tensor(x.tolist()))
    eqx_output = eqx_norm(x)
    
    assert_close(hf_output, eqx_output)

def test_llama_mlp(hf_model, eqx_config):
    _, hf_model = hf_model
    hf_mlp = hf_model.model.layers[0].mlp
    eqx_mlp = LlamaMLP(eqx_config.hidden_size, eqx_config.intermediate_size)
    
    # Copy weights
    eqx_mlp = eqx.tree_at(lambda t: t.gate_proj.weight, eqx_mlp, torch_to_jax(hf_mlp.gate_proj.weight))
    eqx_mlp = eqx.tree_at(lambda t: t.up_proj.weight, eqx_mlp, torch_to_jax(hf_mlp.up_proj.weight))
    eqx_mlp = eqx.tree_at(lambda t: t.down_proj.weight, eqx_mlp, torch_to_jax(hf_mlp.down_proj.weight))
    
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
    eqx_attn = eqx.tree_at(lambda t: t.q_proj.weight, eqx_attn, torch_to_jax(hf_attn.q_proj.weight))
    eqx_attn = eqx.tree_at(lambda t: t.k_proj.weight, eqx_attn, torch_to_jax(hf_attn.k_proj.weight))
    eqx_attn = eqx.tree_at(lambda t: t.v_proj.weight, eqx_attn, torch_to_jax(hf_attn.v_proj.weight))
    eqx_attn = eqx.tree_at(lambda t: t.o_proj.weight, eqx_attn, torch_to_jax(hf_attn.o_proj.weight))
    
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
    eqx_layer = eqx.tree_at(lambda t: t.self_attn.q_proj.weight, eqx_layer, torch_to_jax(hf_layer.self_attn.q_proj.weight))
    eqx_layer = eqx.tree_at(lambda t: t.self_attn.k_proj.weight, eqx_layer, torch_to_jax(hf_layer.self_attn.k_proj.weight))
    eqx_layer = eqx.tree_at(lambda t: t.self_attn.v_proj.weight, eqx_layer, torch_to_jax(hf_layer.self_attn.v_proj.weight))
    eqx_layer = eqx.tree_at(lambda t: t.self_attn.o_proj.weight, eqx_layer, torch_to_jax(hf_layer.self_attn.o_proj.weight))
    eqx_layer = eqx.tree_at(lambda t: t.mlp.gate_proj.weight, eqx_layer, torch_to_jax(hf_layer.mlp.gate_proj.weight))
    eqx_layer = eqx.tree_at(lambda t: t.mlp.up_proj.weight, eqx_layer, torch_to_jax(hf_layer.mlp.up_proj.weight))
    eqx_layer = eqx.tree_at(lambda t: t.mlp.down_proj.weight, eqx_layer, torch_to_jax(hf_layer.mlp.down_proj.weight))
    eqx_layer = eqx.tree_at(lambda t: t.input_layernorm.weight, eqx_layer, torch_to_jax(hf_layer.input_layernorm.weight))
    eqx_layer = eqx.tree_at(lambda t: t.post_attention_layernorm.weight, eqx_layer, torch_to_jax(hf_layer.post_attention_layernorm.weight))
    
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, eqx_config.hidden_size))
    position_ids = jnp.arange(5)[None, :]
    
    hf_output = hf_layer(torch.tensor(x.tolist()), position_ids=torch.tensor(position_ids.tolist()))[0]
    eqx_output = eqx_layer(x, position_ids=position_ids)
    
    assert_close(hf_output, eqx_output)

def test_llama_model(hf_model, eqx_config):
    tokenizer, hf_model = hf_model
    eqx_model = LlamaModel(eqx_config)
    
    # Copy weights (this is a simplified version, you might need to implement a more thorough weight copying function)
    eqx_model = eqx.tree_at(lambda t: t.embed_tokens.weight, eqx_model, torch_to_jax(hf_model.model.embed_tokens.weight))
    eqx_model = eqx.tree_at(lambda t: t.norm.weight, eqx_model, torch_to_jax(hf_model.model.norm.weight))
    for i, layer in enumerate(eqx_model.layers):
        hf_layer = hf_model.model.layers[i]
        eqx_model = eqx.tree_at(lambda t: t.layers[i].self_attn.q_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.q_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].self_attn.k_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.k_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].self_attn.v_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.v_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].self_attn.o_proj.weight, eqx_model, torch_to_jax(hf_layer.self_attn.o_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].mlp.gate_proj.weight, eqx_model, torch_to_jax(hf_layer.mlp.gate_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].mlp.up_proj.weight, eqx_model, torch_to_jax(hf_layer.mlp.up_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].mlp.down_proj.weight, eqx_model, torch_to_jax(hf_layer.mlp.down_proj.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].input_layernorm.weight, eqx_model, torch_to_jax(hf_layer.input_layernorm.weight))
        eqx_model = eqx.tree_at(lambda t: t.layers[i].post_attention_layernorm.weight, eqx_model, torch_to_jax(hf_layer.post_attention_layernorm.weight))
    
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
    
    hf_output = hf_model(input_ids, position_ids=position_ids).logits
    eqx_output = eqx_model(jnp.array(input_ids), position_ids=jnp.array(position_ids))
    
    assert_close(hf_output, eqx_output)
