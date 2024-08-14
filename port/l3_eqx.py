"""
l3_eqx.py

This file contains the Equinox (JAX-based) implementation of the Llama model components.
It is part of a project to convert the PyTorch Llama model to an equivalent JAX/Equinox version.

The module includes implementations of key components of the Llama architecture:

1. LlamaEmbedding: Token embedding layer
2. LlamaLinear: Linear transformation layer
3. LlamaRotaryEmbedding: Rotary position embedding
4. LlamaRMSNorm: Root Mean Square Layer Normalization
5. LlamaSdpaAttention: Scaled Dot-Product Attention mechanism
6. LlamaMLP: Multi-Layer Perceptron
7. LlamaDecoderLayer: A single decoder layer
8. LlamaModel: The main Llama model
9. LlamaForCausalLM: Llama model for causal language modeling
10. LlamaConfig: Configuration class for Llama model

Each class is implemented as an Equinox Module, designed to closely mirror
the functionality of its PyTorch counterpart while leveraging JAX for improved
performance and compatibility with JAX-based workflows.

This file serves as the core of the Equinox Llama implementation, providing
the building blocks for constructing the full model architecture.

Usage:
    This file is imported and used by other scripts to construct and run
    the Equinox version of the Llama model.

Note:
    The implementation now includes the full model structure, including
    the MLP, decoder layer, and the main model classes.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

class LlamaEmbedding(eqx.Module):
    weight: jnp.ndarray

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = jax.random.normal(jax.random.PRNGKey(0), (num_embeddings, embedding_dim))

    def __call__(self, x):
        return jnp.take(self.weight, x, axis=0)

class LlamaLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray | None

    def __init__(self, in_features, out_features, bias=False):
        self.weight = jax.random.normal(jax.random.PRNGKey(0), (out_features, in_features))
        self.bias = jax.random.normal(jax.random.PRNGKey(1), (out_features,)) if bias else None

    def __call__(self, x):
        y = jnp.dot(x, self.weight.T)
        if self.bias is not None:
            y += self.bias
        return y

class LlamaRotaryEmbedding(eqx.Module):
    inv_freq: jnp.ndarray
    max_seq_len_cached: int

    def __init__(self, dim, max_position_embeddings=8192, base=10000):
        self.inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
        self.max_seq_len_cached = max_position_embeddings

    def __call__(self, q, k, seq_len):
        t = jnp.arange(seq_len)
        freqs = jnp.einsum('i,j->ij', t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        
        q_rot = jnp.stack([-q[..., 1::2], q[..., ::2]], axis=-1).reshape(q.shape)
        k_rot = jnp.stack([-k[..., 1::2], k[..., ::2]], axis=-1).reshape(k.shape)
        
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        
        return q, k

class LlamaRMSNorm(eqx.Module):
    weight: jnp.ndarray
    eps: float

    def __init__(self, hidden_size, eps=1e-6):
        self.weight = jnp.ones(hidden_size)
        self.eps = eps

    def __call__(self, hidden_states):
        variance = jnp.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return self.weight * hidden_states

class LlamaSdpaAttention(eqx.Module):
    q_proj: LlamaLinear
    k_proj: LlamaLinear
    v_proj: LlamaLinear
    o_proj: LlamaLinear
    rotary_emb: LlamaRotaryEmbedding
    num_heads: int
    head_dim: int
    
    def __init__(self, hidden_size, num_heads, num_key_value_heads, max_position_embeddings=8192):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = LlamaLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = LlamaLinear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = LlamaLinear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = LlamaLinear(hidden_size, hidden_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings)
    
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, key=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Compute attention
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query_states, key_states)
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = jnp.where(attention_mask[:, None, None, :], attn_weights, float('-inf'))
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value_states)
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class LlamaMLP(eqx.Module):
    gate_proj: LlamaLinear
    up_proj: LlamaLinear
    down_proj: LlamaLinear
    
    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = LlamaLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = LlamaLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = LlamaLinear(intermediate_size, hidden_size, bias=False)
    
    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(eqx.Module):
    self_attn: LlamaSdpaAttention
    mlp: LlamaMLP
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm
    
    def __init__(self, config):
        self.self_attn = LlamaSdpaAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.num_key_value_heads, 
            config.max_position_embeddings
        )
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class LlamaModel(eqx.Module):
    embed_tokens: LlamaEmbedding
    layers: list[LlamaDecoderLayer]
    norm: LlamaRMSNorm
    
    def __init__(self, config):
        self.embed_tokens = LlamaEmbedding(config.vocab_size, config.hidden_size)
        self.layers = [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class LlamaForCausalLM(eqx.Module):
    model: LlamaModel
    lm_head: LlamaLinear
    
    def __init__(self, config):
        self.model = LlamaModel(config)
        self.lm_head = LlamaLinear(config.hidden_size, config.vocab_size, bias=False)
    
    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        return logits

# Configuration class (you might want to expand this based on your needs)
class LlamaConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.intermediate_size = kwargs.get("intermediate_size", 11008)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 32)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 2048)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
