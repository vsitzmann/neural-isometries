from functools import partial
from typing import Tuple, Any, Optional
import jax 
import jax.numpy as jnp
from jax import lax
from jax import custom_jvp
from jax._src.typing import Array 
from jax._src.config import config 
from jax._src.numpy import ufuncs
from flax import linen as nn

EPS = 1.0e-8

def _T(x: Array) -> Array: return jnp.swapaxes(x, -1, -2)
def _H(x: Array) -> Array: return ufuncs.conj(_T(x))

def safe_inverse(x):
  return x / (x**2 + EPS) 


@custom_jvp
def safe_svd(A):

  U, S, VT = jnp.linalg.svd(A, full_matrices=False)
  
  return U, S, VT
  
@safe_svd.defjvp
def safe_svd_jvp(primals, tangents):
  
  A, = primals 
  dA, = tangents 
  
  U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
  
  Ut, V = _H(U), _H(Vt)
  s_dim = s[..., None, :]
  dS = Ut @ dA @ V
  ds = ufuncs.real(jnp.diagonal(dS, 0, -2, -1))



  s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
  s_diffs_zeros = jnp.eye(s.shape[-1], dtype=s.dtype)  # jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
  s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
  #F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
  F = safe_inverse(s_diffs + s_diffs_zeros) - s_diffs_zeros 
  dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
  SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

  s_zeros = (s == 0).astype(s.dtype)
  #s_inv = 1 / (s + s_zeros) - s_zeros
  s_inv = safe_inverse(s + s_zeros) - s_zeros
  s_inv_mat = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(s_inv)
  dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
  dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
  dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))

  m, n = A.shape[-2:]
  if m > n:
    dAV = dA @ V
    dU = dU + (dAV - U @ (Ut @ dAV)) / s_dim.astype(A.dtype)
  if n > m:
    dAHU = _H(dA) @ U
    dV = dV + (dAHU - V @ (Vt @ dAHU)) / s_dim.astype(A.dtype)

  return (U, s, Vt), (dU, ds, _H(dV))
  
@custom_jvp
def ortho_det( U ):
  return jnp.linalg.det( U )
 
 
@ortho_det.defjvp
def ortho_det_jvp(primals, tangents):
  x, = primals
  g, = tangents
  
  y = jnp.linalg.det( x )
  z = jnp.einsum( "...ji, ...jk->...ik", y[..., None, None] * x, g )
  
  return y, jnp.trace( z, axis1=-1, axis2=-2 )
 

# Stable and differentiable procrustes projection
def orthogonal_projection_kernel(X, special=True):
  U, _, VH = safe_svd( X )

  if (X.shape[-2] == X.shape[-1] and special):
    VH = VH.at[..., -1, :].set( ortho_det(jnp.einsum("...ij, ...jk -> ...ik", U, VH))[..., None] * VH[..., -1, :])
  
  R = jnp.einsum( "...ij, ...jk -> ...ik", U, VH )
  
  return R 

def eye_like(R):
  if R.ndim > 2:
    new_dims = tuple(range(len(R[..., 0, 0].shape)))
    I = jnp.expand_dims(jnp.eye(R.shape[-1]), axis=new_dims)
    I = jnp.tile(I, R[..., 0, 0].shape + (1, 1))
  else:
    I = jnp.eye(R.shape[-1])

  return I 

def diag_to_mat(D):
  base_shape = D[..., 0].shape

  DMat = jax.vmap(jnp.diag, (0, None), 0)(jnp.reshape(D, (-1, D.shape[-1])), 0)

  DMat = jnp.reshape(DMat, base_shape + (D.shape[-1], D.shape[-1]))

  return DMat 
  
   
def delta_mask(evals):

  mask = jnp.exp(-1.0 * jnp.abs(evals[..., None] - evals[..., None, :]))
  
  return mask 

  
# Learns operator and estimates isometric maps
class operator_iso( nn.Module ):
  """Operator isometry kernel"""

  op_dim    : int  # Rank of operator, equivalent to number of learned eigenvectors + eigenvalues
  clustered_init: bool = False # Method of eigenvalue initalization
  
  @nn.compact
  def __call__( self, A, B ):

    '''
    Input:
    A, B: batch_size x spatial_dim x channels tensor

    Returns:
    tauOmega: batch_size x op_dim x op_dim
             Isometric eigenbasis map taking the projection of A to the projection of B

    Omega: Tuple (Phi, Lambda, M), 
          Phi: spatial_dim x op_dim tensor of learned eigenfunctions
          Lambda: op_dim tensor of learned eigenvalues (ascending)
          M: spatial_dim tensor of learned mass values
    '''   

    spatial_dim = A.shape[-2]
          
    assert self.op_dim <= spatial_dim #Operator rank must be less than or equal to spatial dimension

    # Projects U to closest matrix Phi perserving inner product Phi^T M Phi = M  
    def _project     ( U, M ):
      MR      = jnp.sqrt( M ) # root of mass matrix
      MRI     = 1.0 / MR 

      Phi     = MRI[..., None] * orthogonal_projection_kernel(MR[..., None] * U, False)
      
      return Phi


    # Solves for isometry best taking projection of A to projection of B in the eigenbasis
    # Equation (8) in the paper
    def _iso_solve( A, B, Phi, Lambda, M ):
      LMask    = delta_mask(Lambda)
      PhiTMB   = jnp.einsum( "...ji, ...jk->...ik", Phi[None, ...], M[None, ..., None] * B )
      PhiTMA   = jnp.einsum( "...ji, ...jk->...ik", Phi[None, ...], M[None, ..., None] * A )

      tauOmega = orthogonal_projection_kernel( LMask[None, ...] * jnp.einsum("...ij,...kj->...ik", PhiTMB, PhiTMA) )
      
      return tauOmega

    def _index( Phi, Lambda, ind):

      return Phi[:, ind], Lambda[ind] 


    # Diagonal values of mass matrix    
    M     = self.param("M", nn.initializers.constant(1.0), (spatial_dim, ), jnp.float32)
    M     =  M ** 2 + 1.0e-8       

    # Eigenfunctions
    Phi   = self.param("Phi", nn.initializers.normal(0.01), (spatial_dim, self.op_dim), jnp.float32)
    Phi   = _project(Phi, M)

    # Eigenvalues
    if not self.clustered_init:
      Lambda     = self.param("Lambda", nn.initializers.normal(1.0/self.op_dim), (self.op_dim, ), jnp.float32)
      Lambda     = jnp.cumsum(Lambda**2, axis=-1)
    else:
      Lambda     = self.param("Lambda", nn.initializers.normal(1.0), (self.op_dim, ), jnp.float32)
      Lambda     = Lambda ** 2 

    # Sort in ascending order          
    o_ind    = jnp.argsort(Lambda, axis=-1)

    Phi      = Phi[:, o_ind]
    Lambda   = Lambda[o_ind]

    # Compute isometric map taking projection of A to projection of B
    tauOmega = _iso_solve(A, B, Phi, Lambda, M)

    Omega = (Phi, Lambda, M)

    return tauOmega, Omega 


       
    
    
