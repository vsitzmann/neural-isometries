import numpy as np
import scipy as sp
import scipy.sparse.linalg as sla
import math
import jax
import jax.numpy as jnp
from sklearn.neighbors import NearestNeighbors

EPS = 1.0e-6


def safe_inverse(x):
  
  return x / (x ** 2 + 1.0e-8)
  
def dict_from_cfg( cfg ):
  context = {}
 
  for setting in dir( cfg ):

    if setting.isupper():
      context[setting] = getattr( cfg, setting )

  return context


def zero_offset(x):
  z_mask = jnp.greater(jnp.abs(x), EPS) 

  return x * z_mask + (1.0 - z_mask) * EPS 


def center_crop(x, H, W):

  oH = (x.shape[-3] - H) // 2 
  oW = (x.shape[-2] - W) // 2 

  return x[..., oH:oH+H, oW:oW+W, :] 




def get_toric_eigs( H, W, num_eigs ):

  # Laplacian Stencil
  stencil = np.asarray( [[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]] )
  #stencil = np.asarray([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
  
  I, J = np.meshgrid( np.arange(H), np.arange(W), indexing="ij" )
  V    = np.reshape( np.concatenate((I[..., None], J[..., None]), axis=-1), (-1, 2) )
  ind  = np.reshape( np.arange(V.shape[0]), (H, W) )
  
  rows = []
  cols = []
  vals = [] 

  for i in range( H ):

    for j in range( W ):

      for u in range( 0, 3 ):
        ii = (i + u - 1) % H
        #ii = (i + u - 1)

        for v in range( 0, 3 ):
          jj = (j + v - 1) % W 
          #jj = (j + v - 1) 
          
          rows.append( ind[i, j] )
          cols.append( ind[ii, jj] )
          vals.append( stencil[u, v] )
  
  rows = np.asarray( rows, dtype=np.int32 )
  cols = np.asarray( cols, dtype=np.int32 )
  vals = np.asarray( vals, dtype=np.float64 )
          
  mass = np.ones( H*W, dtype=np.float64 )
  M    = sp.sparse.diags( mass, shape=(H*W, H*W) )
  L    = -1.0 * sp.sparse.coo_matrix( (vals, (rows, cols)), shape=(H * W, H * W) ) + sp.sparse.identity(H * W)*1.0e-8

  #evals, evecs = sla.eigsh( L.tocsc(), num_eigs, M.tocsc(), sigma=1.0e-8 )
  evals, evecs = sp.linalg.eigh(L.tocsc().toarray(), M.tocsc().toarray())
  return evecs.astype( np.float32 ), evals.astype( np.float32 ), mass.astype( np.float32 ), L.todense(order="C")
 
def lift_9d_se3(rep):
  
  def _normalize(v):
    return v / jnp.sqrt(jnp.clip(jnp.sum(v * v, axis=-1, keepdims=True), a_min=1.0e-8))
    
  n1 = rep[..., :3]
  n2 = rep[..., 3:6]
  v = rep[..., 6:]
  
  x = _normalize(n1)
  z = jnp.cross(x, n2)
  z = _normalize(z) 
  y = jnp.cross(z, x)
  
  R = jnp.concatenate((x[..., None], y[..., None], z[..., None]), axis=-1)
  
  R = jnp.concatenate((R, jnp.zeros_like(R[..., 0, None, :])), axis=-2)
  
  V = jnp.concatenate((v, jnp.ones_like(v[..., 0, None])), axis=-1)
  
  return jnp.concatenate((R, V[..., None]), axis=-1)
  
def inv_se3(g):
  
  gI = jnp.zeros_like(g)
  
  gI = gI.at[..., :3, :3].set(jnp.swapaxes(g[..., :3, :3], -2, -1))
  gI = gI.at[..., :3, -1].set(-1.0 * jnp.matmul( jnp.swapaxes(g[..., :3, :3], -2, -1), g[..., :3, -1, None])[..., 0])
  gI = gI.at[..., -1, -1].set(g[..., -1, -1])
  
  return gI
  

  

