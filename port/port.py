"""
port.py

This script analyzes a Meta-Llama model, printing unique module types and their implementations.
It provides a concise overview of the model's architecture without redundancy.

Key features:
1. Loads the PyTorch Llama model using Hugging Face Transformers.
2. Prints detailed information and implementation for each unique module type.
3. Provides a summary of the model's structure and parameter count.
Usage:
    Run this script to get a comprehensive, non-redundant breakdown of the Llama model architecture.
Note:
    Part of a project to port the Llama model to Equinox/JAX for improved performance.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import inspect
from collections import defaultdict

def load_model(model_name="meta-llama/Meta-Llama-3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.float32,
                                                device_map="cpu",
                                                low_cpu_mem_usage=True)
    model.eval()
    return tokenizer, model

def print_module_info(module_type, module, count):
    print(f"\n{module_type.__name__} (occurs {count} times):")
    print(f"  Example path: {module.example_path}")
    if hasattr(module, 'weight'):
        print(f"  Weight shape: {module.weight.shape}")
    if hasattr(module, 'bias') and module.bias is not None:
        print(f"  Bias shape: {module.bias.shape}")
    if hasattr(module, 'in_features'):
        print(f"  Input features: {module.in_features}")
    if hasattr(module, 'out_features'):
        print(f"  Output features: {module.out_features}")
    
    print("  Custom attributes:")
    for attr_name in dir(module):
        if not attr_name.startswith('_') and not callable(getattr(module, attr_name)) and attr_name not in ['weight', 'bias', 'in_features', 'out_features']:
            print(f"    {attr_name}: {getattr(module, attr_name)}")
    
    print("\n  Implementation:")
    try:
        print(inspect.getsource(module_type))
    except TypeError:
        print(f"  Could not retrieve source code for {module_type.__name__}. It might be implemented in C++.")
    print("-" * 80)

def analyze_model(model):
    module_types = defaultdict(list)
    for name, module in model.named_modules():
        if len(name) > 0:  # Skip the root module
            module_type = type(module)
            if not module_types[module_type]:
                module.example_path = name
            module_types[module_type].append(name)

    print("Unique Module Types and Their Implementations:")
    for module_type, occurrences in module_types.items():
        print_module_info(module_type, model.get_submodule(occurrences[0]), len(occurrences))

    return module_types

def main():
    tokenizer, model = load_model()

    module_types = analyze_model(model)

    # Print summary
    print("\nModel Summary:")
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of unique module types: {len(module_types)}")
    print("\nModule type occurrences:")
    for module_type, occurrences in module_types.items():
        print(f"  {module_type.__name__}: {len(occurrences)}")
if __name__ == "__main__":
    main()
