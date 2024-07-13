import jax 
import jax.numpy as jnp

from functools import partial
from typing import Tuple, Any
from jax._src.typing import Array 
from flax import linen as nn



class LayerNorm( nn.Module ):

  @nn.compact
  def __call__( self, x ):
    mu     = jnp.mean( x, axis=(1, 2), keepdims=True )
    sigma2 = jnp.var( x, axis=(1, 2), keepdims=True )

    x_norm = (x - mu) / jnp.sqrt(sigma2 + 1.0e-6) 

    scale = self.param( "scale", nn.initializers.ones, (1, 1, 1, x.shape[-1]), jnp.float32 )
    bias  = self.param( "bias", nn.initializers.zeros, (1, 1, 1, x.shape[-1]), jnp.float32 ) 

    return x_norm * scale + bias 


   
class ResBlock( nn.Module ):
  channels    : int
  kernel_size : int
  
  @nn.compact
  def __call__( self, x ):
    x0 = LayerNorm()( x )
    x0 = nn.activation.silu( x0 )
    x0 = nn.Conv( self.channels, (self.kernel_size, self.kernel_size) )( x0 )
    
    x0 = LayerNorm()( x0 )
    x0 = nn.activation.silu( x0 )
    x0 = nn.Conv( self.channels, (self.kernel_size, self.kernel_size) )( x0 )
    
    x = x0 + nn.Dense( self.channels )(x) 
    
    return x 
   
 
def Downsample(x):

  return nn.avg_pool(x, (2, 2), strides=(2, 2)) 
  
def Upsample(x):
  return jax.image.resize(x, shape=(x.shape[0], 2*x.shape[1], 2*x.shape[2], x.shape[3]), method="nearest")
  
  
class ConvEncoder(nn.Module):

  channels: Tuple 
  block_depth: Tuple
  kernel_size: int
  out_dim: int

  def setup(self):

    module_list = []

    module_list.append(nn.Dense(48))
    module_list.append(nn.activation.silu)
    
    for l in range(len(self.channels)):
      
      if (l > 0):
        module_list.append(Downsample)
        
      for j in range(self.block_depth[l]):   
        module_list.append(ResBlock(self.channels[l], self.kernel_size))

    module_list.append(LayerNorm())
    module_list.append(nn.activation.silu)  
    module_list.append(nn.Dense(self.out_dim))
    
    self.module_list = module_list 
  
  @nn.compact
  def __call__(self, x):

    for L in self.module_list:

      x = L(x) 
     
    return x 
    

    
class ConvDecoder(nn.Module):

  channels: Tuple 
  block_depth: Tuple
  kernel_size: int 
  out_dim: int
  
  def setup(self):

    module_list = []
  
    for l in range(len(self.channels)):
      
      if (l > 0):
        module_list.append(Upsample)
        
      for j in range(self.block_depth[l]):   
        module_list.append(ResBlock(self.channels[l], self.kernel_size))

    module_list.append(LayerNorm())
    module_list.append(nn.activation.silu)
    module_list.append(nn.Dense(self.out_dim))
    
    self.module_list = module_list 
   
  @nn.compact
  def __call__(self, x):

    
    for L in self.module_list:

      x = L(x) 
    
    return x 



