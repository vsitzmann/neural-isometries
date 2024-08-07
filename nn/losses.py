from functools import partial
from typing import Tuple, Any
import jax 
import jax.numpy as jnp
from jax._src.typing import Array 
from jax._src.config import config 
import numpy as np

import nn.fmaps as fmaps


def l1_loss(x, x_tilde):
  return jnp.mean( jnp.abs(x - x_tilde))

def multiplicity_loss(Lambda):

  G = fmaps.delta_mask(Lambda)
  
  D = fmaps.diag_to_mat(jnp.sum(G, axis=-1))

  L = D - G  

  norm = jnp.sqrt(jnp.sum( L ** 2, axis=(-2, -1)) + 1.0e-8) / L.shape[-1]
  
  return jnp.mean(norm)
  
def scale_procrustes(A, B):

  num = jnp.trace(jnp.matmul(jnp.swapaxes(B, -2, -1), A), axis1=-2, axis2=-1)
  denom = jnp.trace(jnp.matmul(jnp.swapaxes(B, -2, -1), B), axis1=-2, axis2=-1)
  
  s = num / jnp.clip(denom, a_min=1.0e-8)
  
  s = jax.lax.stop_gradient(s)
  
  return A, s[..., None, None] * B 
   
  
def orientation_loss(A, B):
  A = jnp.reshape(A, (-1, 3, 3))
  B = jnp.reshape(B, (-1, 3, 3))

  lossR = jnp.mean(jnp.abs(A - B))
  return lossR
  
def procrustes_loss(A, B):
  Ap, Bp = scale_procrustes(A, B)
  
  lossT = jnp.mean(jnp.abs(Ap - Bp))
  
  return lossT 
