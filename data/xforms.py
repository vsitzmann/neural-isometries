import jax
import jax.numpy as jnp
import numpy as np
import functools
from utils.utils import zero_offset
from icecream import ic 
from scipy.spatial.transform import Rotation as R

def safe_inverse(x):
  return x / (x **2 + 1.0e-8)
  
'''
=========================================================================
============================== Homographies =============================
=========================================================================
'''

# Adapted from https://github.com/lemacdonald/equivariant-convolutions

# sl(3, R) basis
def hom_basis():

  e1 = jnp.array([1, 0, 0]).astype(jnp.float32)
  e2 = jnp.array([0, 1, 0]).astype(jnp.float32)
  e3 = jnp.array([0, 0, 1]).astype(jnp.float32)

  B1 = jnp.matmul(e1[:, None], e1[None, :]) - (1.0 /3.0) * jnp.eye(3)
  B2 = jnp.matmul(e1[:, None], e2[None, :]) 
  B3 = jnp.matmul(e1[:, None], e3[None, :])
  B4 = jnp.matmul(e2[:, None], e1[None, :])
  B5 = jnp.matmul(e2[:, None], e2[None, :]) - (1.0/3.0) * jnp.eye(3)
  B6 = jnp.matmul(e2[:, None], e3[None, :]) 
  B7 = jnp.matmul(e3[:, None], e1[None, :])
  B8 = jnp.matmul(e3[:, None], e2[None, :]) 

  return jnp.stack([B1, B2, B3, B4, B5, B6, B7, B8], axis=0)

  
  
@functools.partial(jax.jit, static_argnames=['num_xforms'])   
def _random_xforms(basis, scales, key, num_xforms):
  
  key, xform_key, rot_key = jax.random.split(key, 3)

  coeff = jax.random.normal(xform_key, (num_xforms, scales.shape[0])) * 0.5 * scales[None, ...] 
 
  Alg = jnp.sum(coeff[..., None, None] * basis[None, ...], axis=1)
  
  return jax.scipy.linalg.expm(Alg), key

  

@functools.partial(jax.jit, static_argnames=[ 'num_xforms', 'H', 'W'])   
def _draw_and_xform(basis, scales, key, num_xforms, H, W):

  A, key = _random_xforms(basis, scales, key, num_xforms) 

  I, J = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')

  I = 2.0*((jnp.reshape(I, (-1, )) / (H - 1)) - 0.5) 
  J = 2.0*((jnp.reshape(J, (-1, )) / (W - 1)) - 0.5) 

  X = jnp.stack([I, J, jnp.ones_like(I)], axis=-1) 

  AX = jnp.matmul(A[:, None, ...], X[None, ..., None])[..., 0]

  AX = AX[..., :2] * safe_inverse(AX[..., 2, None])   #zero_offset(AX[..., 2, None])

  
  AX = 0.5 * (AX + 1.0)

  AI = AX[..., 0] * (H - 1)
  AJ = AX[..., 1] * (W - 1) 

  
  return AI, AJ, key 


@jax.jit
def interp(img, x, y):

  #x = jnp.round(x)
  #y = jnp.round(y)
  
  x0 = jnp.floor(x).astype(jnp.int32)
  x1 = x0 + 1 
  y0 = jnp.floor(y).astype(jnp.int32)
  y1 = y0 + 1 

  diffx = jnp.abs(x - x0)
  diffy = jnp.abs(y - y0)
  
  
  mask = jnp.logical_and(jnp.logical_and(jax.lax.ge(x, 0.0), jax.lax.ge(img.shape[0]-1.0, x)),
                         jnp.logical_and(jax.lax.ge(y, 0.0), jax.lax.ge(img.shape[1]-1.0, y)))
 
  
  x0 = jnp.clip(x0, 0, img.shape[0]-1)
  x1 = jnp.clip(x1, 0, img.shape[0]-1)
  y0 = jnp.clip(y0, 0, img.shape[1]-1)
  y1 = jnp.clip(y1, 0, img.shape[1]-1)

  Ia = img[x0, y0, :]
  Ib = img[x0, y1, :]
  Ic = img[x1, y0, :]
  Id = img[x1, y1, :]

  wa = (x1-x)*(y1-y)
  wb = (x1-x)*(y-y0)
  wc = (x-x0)*(y1-y)
  wd = (x-x0)*(y-y0)

  itp = mask[:, None] * (wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id) 
  
  return jnp.reshape(itp, img.shape)
  
@jax.jit   
def _draw_and_apply_pair(x, basis, scales, key):


  H = x.shape[1]
  W = x.shape[2]
  C = x.shape[3]  

  AI, AJ, key = _draw_and_xform(basis, scales, key, x.shape[0], H, W) 

  #AI, AJ, key = _draw_and_xform_mobius(key, x.shape[0], H, W)
  
  x_min = jnp.min(x, axis=(1, 2), keepdims=True)

  #print(AI.shape, flush=True)
  #print(x.shape, flush=True)
  
  Ax = jax.vmap(interp, in_axes=(0, 0, 0), out_axes=0)(x - x_min, AI, AJ) + x_min
  
  return jnp.reshape(jnp.concatenate((x[:, None, ...], Ax[:, None, ...]), axis=1), (-1, H, W, C))
  
@functools.partial(jax.jit, static_argnames=['num_xforms'])   
def _draw_and_apply_single(x, basis, scales, key, num_xforms):


  H = x.shape[1]
  W = x.shape[2]
  C = x.shape[3]  
  x = jnp.reshape(x, (num_xforms, -1, x.shape[1], x.shape[2], x.shape[3]))

  AI, AJ, key = _draw_and_xform(basis, scales, key, num_xforms, H, W) 
  
  Ax = jax.vmap(interp, in_axes=(0, 0, 0), out_axes=0)(x, AI, AJ) 
  
  return jnp.reshape(Ax, (-1, H, W, C))




class Homographies():

  def __init__(self):

    self.B = hom_basis() 
    self.scales = 2.0 * jnp.array([0.15, 0.35, 0.17, 0.35, 0.15, 0.17, 0.15, 0.15]).astype(jnp.float32) 
    
  def draw_and_apply(self, x, key):
    return _draw_and_apply_pair(x, self.B, self.scales, key) 

'''
=========================================================================
======================== Mobius Transformations =========================
=========================================================================
'''

def z_rot(ang):

  RZ = jnp.zeros( ang.shape + (3, 3), dtype=jnp.float32)
  c = jnp.cos(ang)
  s = jnp.sin(ang)

  RZ = RZ.at[..., 0, 0].set(c)
  RZ = RZ.at[..., 0, 1].set(-s)
  RZ = RZ.at[..., 1, 0].set(s)
  RZ = RZ.at[..., 1, 1].set(c)
  RZ = RZ.at[..., 2, 2].set(1.0)

  return RZ

def y_rot(ang):
  
  RY = jnp.zeros( ang.shape + (3, 3), dtype=jnp.float32)
  c = jnp.cos(ang)
  s = jnp.sin(ang)

  RY = RY.at[..., 0, 0].set(c)
  RY = RY.at[..., 0, 2].set(s)
  RY = RY.at[..., 2, 0].set(-s)
  RY = RY.at[..., 2, 2].set(c)
  RY = RY.at[..., 1, 1].set(1.0)

  return RY 
  
def zyz_from_euler(angles):

    return jnp.einsum("...ij, ...jk, ...kl->...il", z_rot(angles[..., 0]), y_rot(angles[..., 1]), z_rot(angles[..., 2]))

    
@functools.partial(jax.jit, static_argnames=['shift'])   
def xform_mobius(x, mob_key, shift=0.2):

  H, W, NC = x.shape[1], x.shape[2], x.shape[3]
  
  I, J = jnp.meshgrid(jnp.arange(x.shape[1]), jnp.arange(x.shape[2]), indexing='ij')

  I = jnp.reshape(I, (-1, ))
  J = jnp.reshape(J, (-1, ))

  
  Phi = np.pi * (I / (H-1))
  Theta = 2.0 * np.pi * ( J /(W - 1))

  X1 = jnp.cos(Theta) * jnp.sin(Phi)
  X2 = jnp.sin(Theta) * jnp.sin(Phi)
  X3 = jnp.cos(Phi)

  P0 = jnp.concatenate((X1[..., None], X2[..., None], X3[..., None]), axis=-1)


  pc = jax.random.uniform(mob_key, (x.shape[0], 6))

  pc = pc * jnp.asarray([2.0 * np.pi, np.pi, 2.0 * np.pi, 2.0*np.pi, np.pi, shift])[None, :] 
  #pc = pc * jnp.asarray([2.0 * np.pi, 0.0, 0.0, 2.0*np.pi, np.pi, shift])[None, :] 
  C1 = jnp.cos(pc[:, 3]) * jnp.sin(pc[:, 4]) 
  C2 = jnp.sin(pc[:, 3]) * jnp.sin(pc[:, 4])
  C3 = jnp.cos(pc[:, 4])

  C = pc[:, 5, None] * jnp.concatenate((C1[..., None], C2[..., None], C3[..., None]), axis=-1)

  CT1 = 1.0 - jnp.sum( C ** 2, axis=-1)

  PC = P0[None, ...] + C[:, None, ...] 

  PP = (CT1[:, None, None] * PC / jnp.sum(PC ** 2 , axis=-1, keepdims=True)) + C[:, None, ...]

  PP = PP / jnp.linalg.norm(PP, axis=-1, keepdims=True)

  R =  zyz_from_euler(pc[:, :3])

  #ic(R.shape)
  PP = jnp.matmul(R[:, None, ...], PP[..., None])[..., 0] 
  #PP = jnp.matmul(R[:, None, ...], P0[..., None])[..., 0] 
  ThetaM = jnp.arctan2(PP[..., 1], PP[..., 0])
  PhiM = jnp.arccos(jnp.clip(PP[..., 2], a_min=-1.0, a_max=1.0))

  mask = (ThetaM < 0)
  ThetaM = (1.0 - mask) * ThetaM +  mask * (ThetaM + 2.0 * np.pi)
  
  IM = jnp.clip(PhiM * (H-1) / np.pi, a_min=0, a_max=H-1)
  JM = jnp.clip(ThetaM * (W-1) / (2.0 * np.pi), a_min=0, a_max=W-1)

  x_min = jnp.min(x, axis=(1, 2), keepdims=True)

  #print(AI.shape, flush=True)
  #print(x.shape, flush=True)

 
  
  Mx = jax.vmap(interp, in_axes=(0, 0, 0), out_axes=0)(x - x_min, IM, JM) + x_min
  #ic(Mx.shape)
  #ic(x.shape)
  #Mx = jnp.roll(x, 50, axis=2)
  out = jnp.reshape(jnp.concatenate((x[:, None, ...], Mx[:, None, ...]), axis=1), (-1, H, W, NC))
  
  return out
  

def ExtMobius(x, shift=1.2):
  
  offset = np.mean(x, axis=0, keepdims=True)

  x0 = x - offset 

  r0 = np.max(np.linalg.norm(x0, axis=-1))

  x0 = (x0 / r0) * 0.9 

  pc = np.random.rand(5)


  pc[0] = 2.0 * np.pi * pc[0]
  pc[1] = np.pi * pc[1]
  pc[2] = 2.0 * np.pi * pc[2]
  pc[3] = 2.0 * np.pi * pc[3]
  pc[4] = np.pi * pc[4]

  c = np.asarray([np.cos(pc[3]) * np.sin(pc[4]), np.sin(pc[3]) * np.sin(pc[4]), np.cos(pc[4])])

  c = c * shift 

  CT1 = 1.0 - jnp.sum(c ** 2)

  xc = x0 + c 

  xp = (xc * CT1 / jnp.sum(xc ** 2, axis=-1, keepdims=True)) + c[None, ...]

  M = R.from_euler('ZYZ', [pc[0], pc[1], pc[2]]).as_matrix()
  
  xp = np.matmul(M, xp[..., None])[..., 0]
  
  #xp = (xp * r0 / 0.9) + offset 
  
  xp = xp / 0.9 
  x0 = x0 / 0.9 
  
  return x0, xp
  
def distGC( theta0, phi0, theta, phi):
    
    EPS = 1e-12;
    
    val = np.cos(phi) * np.cos(phi0) + np.cos( theta - theta0) * np.sin(phi) * np.sin(phi0) 
    
    return np.arccos( np.clip(val, a_min=-1.0, a_max=1.0))

    
def estimateMob(p1, p2, nSamples=100):
    
    theta1 = np.arctan2(p1[..., 1], p1[..., 0]);
    theta2 = np.arctan2(p2[..., 1], p2[..., 0]);
    
    phi1 = np.arccos(np.clip(p1[..., 2], a_min=-1.0, a_max=1.0));
    phi2 = np.arccos(np.clip(p2[..., 2], a_min=-1.0, a_max=1.0));
    
    z1 = np.tan(phi1/2.0)*np.exp(-1j*theta1);
    z2 = np.tan(phi2/2.0)*np.exp(-1j*theta2);
    
    maxS = z1.shape[0] // 3;
    
    if (nSamples > maxS):
        nSamples = maxS;
    
    ind = np.random.permutation(z1.shape[0])
    
    idx1 = ind[:nSamples];
    idx2 = ind[nSamples:2*nSamples];
    idx3 = ind[2*nSamples:3*nSamples];
    
    u1 = z1[idx1];
    u2 = z1[idx2];
    u3 = z1[idx3];
    
    v1 = z2[idx1];
    v2 = z2[idx2];
    v3 = z2[idx3];
    
    
    M1 = np.ones( (nSamples, 3, 3), dtype=np.complex64)
    M2 = np.copy(M1);
    M3 = np.copy(M2);
    M4 = np.copy(M3);
    
    C1 = np.zeros( (nSamples, 3), dtype=np.complex64)
    C2 = np.copy(C1)
    C3 = np.copy(C2)
    
    C3[:, 0] = u1 * v1;
    C3[:, 1] = u2 * v2;
    C3[:, 2] = u3 * v3;
    
    C1[:, 0] = u1;
    C1[:, 1] = u2;
    C1[:, 2] = u3;
    
    C2[:, 0] = v1;
    C2[:, 1] = v2;
    C2[:, 2] = v3;
    
    
    M1[:, :, 0] = C3;
    M1[:, :, 1] = C2;
    
    M2[:, :, 0] = C3;
    M2[:, :, 1] = C1;
    M2[:, :, 2] = C2;
    
    M3[:, :, 0] = C1;
    M3[:, :, 1] = C2;
    
    M4[:, :, 0] = C3;
    M4[:, :, 1] = C1;
    
    a = np.linalg.det(M1);
    b = np.linalg.det(M2);
    c = np.linalg.det(M3);
    d = np.linalg.det(M4);
    
    z2p = (a[None, :] * z1[:, None] + b[None, :])/(c[None, :] * z1[:, None] + d[None, :]);
    
    theta2p = np.arctan2(-1.0*np.imag(z2p), np.real(z2p));
    phi2p = 2*np.arctan(np.abs(z2p))
    
    d = distGC(theta2[:, None], phi2[:, None], theta2p, phi2p) 
    
    dM = np.nanmean(d)
    
    return dM

def estimate_mobius(p1, p2):

  theta1 = np.arctan2(p1[..., 1], p1[..., 0]);
  theta2 = np.arctan2(p2[..., 1], p2[..., 0]);
  
  phi1 = np.arccos(np.clip(p1[..., 2], a_min=-1.0, a_max=1.0));
  phi2 = np.arccos(np.clip(p2[..., 2], a_min=-1.0, a_max=1.0));

  null_mask = np.logical_or(phi1/2.0 > 1.56, phi2/2.0 > 1.56)
  phi1 = phi1 * (1.0 - null_mask)
  phi2 = phi2 * (1.0 - null_mask)
  
  z1 = np.tan(phi1/2.0)*np.exp(-1j*theta1);
  z2 = np.tan(phi2/2.0)*np.exp(-1j*theta2);

  P = np.concatenate((-z1[..., None], -1.0 * np.ones_like(z1[..., None]), (z1 * z2)[..., None], z2[..., None]), axis=-1)


  _, _, VH = np.linalg.svd(P, full_matrices=False)

  V = np.conjugate(np.swapaxes(VH, -2, -1))

  mob_vec = V[..., -1]
  mob_xform = np.concatenate((mob_vec[None, :2], mob_vec[None, 2:]), axis=-2)

  mob_xform = mob_xform / np.sqrt(np.linalg.det(mob_xform))

  #ic(np.linalg.det(mob_xform))

  z2p = (mob_xform[..., 0, 0] * z1 + mob_xform[..., 0, 1])/(mob_xform[..., 1, 0] * z1 + mob_xform[..., 1, 1] );
  
  theta2p = np.arctan2(-1.0*np.imag(z2p), np.real(z2p));
  phi2p = 2*np.arctan(np.abs(z2p))
  
  d = distGC(theta2, phi2, theta2p, phi2p) 
  
  return mob_xform, np.mean(d)


  
@functools.partial(jax.jit, static_argnames=['base_weight'])   
def draw_shrec11_pairs(batch, key, base_weight=0.2):

  H, W, C = batch.shape[-3], batch.shape[-2], batch.shape[-1]
  
  def _extract_pair(batch, ind):
    p0 = batch[ind[0], ...]
    p1 = batch[ind[1], ...]

    return jnp.concatenate((p0[None, ...], p1[None, ...]), axis=0)

  if base_weight is None:
    pair_inds = jax.random.choice(key, batch.shape[1], (batch.shape[0], 2))
  else:
    p0_key, p1_key = jax.random.split(key)
    p = jnp.ones(batch.shape[1])
    p = p.at[0].set(base_weight) 
    p = p.at[1:].set( (1.0 - base_weight) / (batch.shape[1] - 1))

    ind0 = jax.random.choice(p0_key, batch.shape[1], (batch.shape[0], 1), p=p)
    ind1 = jax.random.choice(p1_key, batch.shape[1], (batch.shape[0], 1))

    pair_inds = jnp.concatenate((ind0, ind1), axis=-1)
    
  batch_pairs = jax.vmap(_extract_pair, (0, 0), 0)(batch, pair_inds)

  return jnp.reshape(batch_pairs, (-1, H, W, C))


@jax.jit 
def draw_shrec11(batch, key):
  
  def _extract(batch, ind):
    return batch[ind, ...]

  ind = jax.random.choice(key, batch.shape[1], (batch.shape[0], ))

  batch = jax.vmap(_extract, (0, 0), 0)(batch, ind)
  
  return batch
  

  
  

  
  