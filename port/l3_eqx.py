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

    def __init__(self, config):
        dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len_cached = config.max_position_embeddings
        inv_freq = 1.0 / (config.rope_theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, x, position_ids):
        seq_len = position_ids.shape[1]
        t = position_ids.astype(jnp.float32)
        inv_freq = self.inv_freq

        # Reshape t to match the expected input shape
        t = t.reshape(-1, seq_len, 1)  # Shape: (batch_size, seq_len, 1)
        
        # Compute freqs directly without using einsum
        freqs = t * inv_freq[None, None, :]  # Shape: (batch_size, seq_len, dim//2)
        
        emb = jnp.concatenate((freqs, freqs), axis=-1)  # Shape: (batch_size, seq_len, dim)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)

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
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    rope_theta: float

    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = LlamaLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = LlamaLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = LlamaLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = LlamaLinear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, key=None):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Implement caching logic here if needed
            pass

        if self.num_key_value_heads != self.num_heads:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query_states, key_states) / jnp.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = jnp.where(attention_mask[:, None, None, :kv_seq_len], attn_weights, jnp.finfo(attn_weights.dtype).min)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    bs, n_kv_head, seqlen, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.repeat(x[:, None, :, :, :], n_rep, axis=1).reshape(bs, n_kv_head * n_rep, seqlen, head_dim)

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
        
        # New attributes
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.attention_bias = kwargs.get("attention_bias", False)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.use_cache = kwargs.get("use_cache", True)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        
        # Derived attributes
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        
        # Optional attributes
        self.bias = kwargs.get("bias", False)  # For compatibility with some attention implementations
        self.rope_type = kwargs.get("rope_type", "default")
        self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 1.0)
        
        # Dropout rates (usually 0.0 for inference)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.hidden_dropout = kwargs.get("hidden_dropout", 0.0)
        
        # Additional optional parameters
        self.bos_token_id = kwargs.get("bos_token_id", None)
        self.eos_token_id = kwargs.get("eos_token_id", None)
        self.pad_token_id = kwargs.get("pad_token_id", None)
        self.torch_dtype = kwargs.get("torch_dtype", None)
        
    def __repr__(self):
        return f"LlamaConfig({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"
