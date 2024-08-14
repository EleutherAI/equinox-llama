"""
test_jax_torch.py

This script serves as a compatibility test to ensure that both PyTorch and JAX can be used
simultaneously within the same Python environment. It performs the following tasks:

1. Imports both PyTorch and JAX libraries.
2. Defines simple functions using PyTorch tensors and JAX arrays.
3. Executes these functions to perform basic computations.
4. Prints the results and device information for both libraries.

The primary purpose of this script is to verify that:
- Both PyTorch and JAX are correctly installed.
- There are no conflicts between the two libraries.
- Basic operations can be performed using both frameworks.
- The script can identify and utilize available hardware (CPU/GPU) for both libraries.

This test is crucial for the larger project of porting Llama models from PyTorch to JAX/Equinox,
as it confirms that the development environment is properly set up to work with both frameworks.

Usage:
    Run this script to check if PyTorch and JAX are correctly installed and can operate together.
    The output will show computation results and device information for both libraries.

Note:
    If this script runs successfully, it indicates that the environment is ready for the model
    conversion process. Any errors or inconsistencies should be addressed before proceeding with
    the main porting tasks.
"""

import torch
import jax
import jax.numpy as jnp

def torch_function():
    # Create a PyTorch tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    return torch.dot(x, y)

def jax_function():
    # Create a JAX array
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    return jnp.dot(x, y)

def main():
    print("Testing PyTorch:")
    torch_result = torch_function()
    print(f"PyTorch result: {torch_result}")
    print(f"PyTorch device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("\nTesting JAX:")
    jax_result = jax_function()
    print(f"JAX result: {jax_result}")
    print(f"JAX devices: {jax.devices()}")

    print("\nBoth libraries are working correctly if you see results above.")

if __name__ == "__main__":
    main()
