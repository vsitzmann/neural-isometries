from functools import partial
from typing import Tuple, Any
import jax 
import jax.numpy as jnp
from jax._src.typing import Array 
from flax import linen as nn



class MLP(nn.Module):
  features: int 
  out_dim: int
  num_layers: int = 1 
  skip_init_act: bool = False 
  
  def setup(self):
    
    layers = []
    for l in range(self.num_layers):
    
      if (l != 0 or not self.skip_init_act):
        layers.append(jax.nn.silu)
        
      if (l == self.num_layers - 1):
        F = self.out_dim
      else:
        F = self.features 
        
      layers.append(nn.Dense(features=F)) 

    
    self.layers = layers 
    

  def __call__(self, x: Array):
  
    for L in self.layers:
      x = L(x)
    
    return x 


class skipMLP(nn.Module):
  features: int
  out_dim: int 
  num_layers: int 
  skip_init_act: bool = False
  
  @nn.compact
  def __call__(self, x: Array):
    
    N1 = (self.num_layers // 2) + (self.num_layers % 2) 
    N2 = self.num_layers - N1 
      
    x0 = MLP(features=self.features, out_dim=self.features, num_layers=N1, skip_init_act=self.skip_init_act)(x)
    
    x0 = x0 + nn.Dense(features=self.features)(x) 
    
    x1 = MLP(features=self.features, out_dim=self.out_dim, num_layers=N2, skip_init_act=False)(x0)
    
    return x1 
  
