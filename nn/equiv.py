from functools import partial
from typing import Tuple, Any
import jax 
import jax.numpy as jnp
from jax._src.typing import Array 
#from jax.config import config 
from flax import linen as nn

EPS = 1.0e-6

def safe_inverse(x):
  
  return x / (x * x + EPS) 
  
  
class orth_neuron(nn.Module):
  
  features: int 
  
  @nn.compact
  def __call__(self, x: Array):
    # x: B x K x d 
    
    q = nn.Dense(features=self.features, use_bias=False)(x)
    k = nn.Dense(features=self.features, use_bias=False)(x) 
    
    dot = jnp.sum(q * k, axis=-2, keepdims=True)
    
    k2 = jnp.sum(k * k, axis=-2, keepdims=True) 
    
    k = jax.nn.silu(-1.0 * dot) * safe_inverse(k2) * k 
    
    return q + k 
    
 
class orth_norm(nn.Module):
  
  @nn.compact
  def __call__(self, x: Array): 
  # x: B x K x d 
  
    features = x.shape[-1]
    
    eps = self.param("eps", nn.initializers.constant(1.0e-3), (features, ), jnp.float32)
    scale = self.param("scale", nn.initializers.ones, (features, ), jnp.float32) 
    
    x2 = (jnp.sum(x * x, axis=-2, keepdims=True) / x.shape[-2])
     
    factor = scale[None, None, :] * jax.lax.rsqrt(x2 + EPS + jnp.abs(eps)[None, None, :]) 
    
    return x * factor 
  
 
  
class orth_mlp(nn.Module):

  features: int 
  num_layers: int = 1 
  factor: int = 1 
  
  def setup(self):
    
    layers = []
    for l in range(self.num_layers):
    
      if (l == self.num_layers - 1):
        F = self.features 
      else:
        F = self.features * self.factor 
        
      layers.append(orth_norm())
      layers.append(orth_neuron(features=F)) 
    
    self.layers = layers 
    

  def __call__(self, x: Array):
  
    for l in range(self.num_layers):
      x = self.layers[l](x)
    
    return x 


 
class orth_out(nn.Module):

  out_dim: int
  inter_features: int = 64
  
  
  @nn.compact
  def __call__(self, x: Array):
    
    # x: B x K x d 
    x = orth_norm()(x)
    x = orth_neuron(features=self.inter_features)(x)
    
    irows, icols = jnp.triu_indices(self.inter_features, k=0)
    x = jnp.matmul(jnp.swapaxes(x, -2, -1), x)[:, irows, icols]
    
    x = nn.Dense(features=4*x.shape[-1])(x)
    x = jax.nn.silu(x) 
    x = nn.Dense(features=self.out_dim)(x)
    
    return x 

class orth_net(nn.Module):
  features: int
  num_layers: int  
  out_dim: int
  
  @nn.compact
  def __call__(self, x: Array):
  
    x0 = orth_mlp(features=self.features, num_layers=(self.num_layers//2)+1)(x) 
    x0 = x0 + nn.Dense(features=self.features, use_bias=False)(x) 
    
    x1 = orth_mlp(features=self.features, num_layers=(self.num_layers//2)+1)(x0)
    
    return orth_out(out_dim=self.out_dim)(x1)

