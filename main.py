import jax
import jax.numpy as jnp

grad_tanh = jax.grad(jnp.tanh)
print(grad_tanh(0.5))  