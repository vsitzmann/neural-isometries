"""
Train.

Usage:
  train [options]
  train (-h | --help)
  train --version

Options:
  -h --help               Show this screen.
  --version               Show version.
  -i, --in=<input_dir>    Input directory  [default: {root_path}/data/CSHREC_11/processed/]. 
  -o, --out=<output_dir>  Output directory [default: {root_path}/experiments/cshrec11_encode/weights/].
  -c, --config=<config>   Config directory [default: config].
"""

import os
import sys
import functools
import importlib.util
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds 
import matplotlib
import matplotlib.cm as cm 
import wandb
from docopt import docopt

print(jax.default_backend(), flush=True)
tf.config.experimental.set_visible_devices( [], 'GPU' )

from typing import Any, Callable, Dict, Sequence, Tuple, Union
from clu import checkpoint, metric_writers, metrics, parameter_overview, periodic_actions
from tqdm import tqdm
from icecream import ic

from os.path import dirname, abspath
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.append( ROOT_PATH )

from nn import losses, conv, fmaps
from utils import utils
from utils.color import rwb_map, rwb_thick_map
from data import data_loader as dl
from data import input_pipeline, xforms
from utils import ioutils as io


# Constants
PMAP_AXIS = "batch" 

clr_norm   = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
clr_mapper_K = cm.ScalarMappable(norm=clr_norm, cmap=rwb_map)
clr_mapper_T = cm.ScalarMappable(norm=clr_norm, cmap=rwb_thick_map)

@flax.struct.dataclass
class TrainState:
  step        : int
  opt_state   : Any
  params      : Any
  key         : Any 
  multi_steps : Any 


def create_train_state( cfg: Any, data_shape: Tuple ) -> Tuple[nn.Module, nn.Module, Any, Any, Any, TrainState]:

  # Random key 
  seed = 0 #np.random.randint(low=0, high=1e8, size=(1, ))[0]
  
  key                                          = jax.random.PRNGKey( seed )
  key, enc_key, dec_key, kernel_key, xform_key = jax.random.split( key, 5 )
  
  down_factor = np.power( 2, (len(cfg.CONV_ENC_CHANNELS) - 1) )
  
      
  # Models

  encoder = conv.ConvEncoder( channels    = cfg.CONV_ENC_CHANNELS,
                              block_depth = cfg.CONV_ENC_BLOCK_DEPTH,
                              kernel_size = cfg.KERNEL_SIZE,
                              out_dim     = cfg.LATENT_DIM)
                              
  decoder = conv.ConvDecoder( channels    = cfg.CONV_DEC_CHANNELS,
                              block_depth = cfg.CONV_DEC_BLOCK_DEPTH,
                              kernel_size = cfg.KERNEL_SIZE,
                              out_dim     = data_shape[-1])
  
                       
  dec_input = jnp.ones( (data_shape[0], data_shape[1] // down_factor, data_shape[2] // down_factor, cfg.LATENT_DIM), dtype=jnp.float32 )

      
  kernel_input = jnp.reshape( dec_input, (dec_input.shape[0], -1,  cfg.LATENT_DIM) )
            
  kernel = fmaps.operator_iso(op_dim=cfg.KERNEL_OP_DIM)
  
  # Initialize
  enc_input     = jnp.ones( data_shape, dtype=jnp.float32 )
  enc_params    = encoder.init( enc_key, enc_input )["params"]
  dec_params    = decoder.init( dec_key, dec_input )["params"]
  kernel_params = kernel.init( kernel_key, kernel_input, kernel_input )["params"]
  params        = {"encoder": enc_params, "decoder": dec_params, "kernel": kernel_params} 

  # Set up optimizer 
  schedule = optax.warmup_cosine_decay_schedule( init_value   = cfg.INIT_LR,
                                                 peak_value   = cfg.LR,
                                                 warmup_steps = cfg.WARMUP_STEPS,
                                                 decay_steps  = cfg.NUM_TRAIN_STEPS,
                                                 end_value    = cfg.END_LR )

  batch_size  = cfg.BATCH_SIZE
  multi_steps = cfg.TRUE_BATCH_SIZE // batch_size
  
  optim = optax.adamw( learning_rate=schedule, b1=cfg.ADAM_B1, b2=cfg.ADAM_B2 )

  optimizer = optax.MultiSteps( optim, multi_steps )
  
  
  state       = optimizer.init( params ) 
  train_state = TrainState( step=0, opt_state=state, params=params, key=key, multi_steps=multi_steps )
  
  return encoder, decoder, kernel, optimizer, xform_key, train_state


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  train_loss        : metrics.Average.from_output("train_loss")
  train_recon_loss    : metrics.Average.from_output("train_recon_loss")
  train_equiv_loss  : metrics.Average.from_output("train_equiv_loss") 
  train_mult_loss  : metrics.Average.from_output("train_mult_loss") 

@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  eval_loss         : metrics.Average.from_output("eval_loss")
  eval_recon_loss     : metrics.Average.from_output("eval_recon_loss")
  eval_equiv_loss   : metrics.Average.from_output("eval_equiv_loss") 
  eval_mult_loss   : metrics.Average.from_output("eval_mult_loss") 


def train_step(x: Any, encoder: nn.Module, decoder: nn.Module, kernel: Any, state: TrainState,
               optimizer: Any, alpha_equiv: float, beta_mult: float, train: bool = True):
  
  key    = state.key 
  step   = state.step+1
  
  
  def loss_fn(params):
     
    z = encoder.apply({"params": params["encoder"]}, x)
   
    zH, zW, latent_dim = z.shape[1], z.shape[2], z.shape[3]
  
    z       = jnp.reshape(z, (-1, 2, zH, zW, latent_dim))
  
    z0      = jnp.reshape(z[:, 0, ...], (z.shape[0], -1, latent_dim))
    Tz0     = jnp.reshape(z[:, 1, ...], (z.shape[0], -1, latent_dim))
  
    tauOmegaBA, Omega  = kernel.apply({"params": params["kernel"]}, z0, Tz0)
    tauOmegaAB         = jnp.swapaxes(tauOmegaBA, -2, -1)    

    # Project and transform
    Tz0_p = jnp.einsum( "...jk, ...lk, ...lm->...jm", tauOmegaBA, Omega[0][None, ...], Omega[2][None, ..., None] * z0 )
    z0_p  = jnp.einsum( "...jk, ...lk, ...lm->...jm", tauOmegaAB, Omega[0][None, ...], Omega[2][None, ..., None] * Tz0 )

    # Project 
    z0_b  = jnp.einsum( "...lj, ...lm->...jm", Omega[0][None, ...], Omega[2][None, ..., None] * z0 )
    Tz0_b = jnp.einsum( "...lj, ...lm->...jm", Omega[0][None, ...], Omega[2][None, ..., None] * Tz0 )
    
    
    # Multiplicity and equivariance losses 
    loss_mult  = losses.multiplicity_loss(Omega[1])
        
    loss_equiv = losses.l1_loss(Tz0_b, Tz0_p) + losses.l1_loss(z0_b, z0_p) 

    # Unproject
    Tz0_p = jnp.einsum( "...ij, ...jk->...ik", Omega[0][None, ...], Tz0_p )
    z0_p  = jnp.einsum( "...ij, ...jk->...ik", Omega[0][None, ...], z0_p )
    Tz0_b = jnp.einsum( "...ij, ...jk->...ik", Omega[0][None, ...], Tz0_b )
    z0_b  = jnp.einsum( "...ij, ...jk->...ik", Omega[0][None, ...], z0_b )
    
   # Aggregate latents for decoding 
    z_p      = jnp.concatenate((z0_b[:, None, ...], Tz0_b[:, None, ...], z0_p[:, None, ...], Tz0_p[:, None, ...]), axis=1)
    z_p      = jnp.reshape(z_p, (-1, zH * zW, latent_dim))
    z_p      = jnp.reshape(z_p, (z_p.shape[0], zH, zW, latent_dim))     
    
    

    # Decode 
    x_out     = decoder.apply({"params":params["decoder"]}, z_p)
    
    x_out     = jnp.reshape(x_out, (-1, 4, x.shape[1], x.shape[2], x.shape[3]))
    
    x0b_p     = x_out[:, 2, ...]
    Tx0b_p    = x_out[:, 3, ...]
    
    xR        = jnp.reshape(x, (-1, 2, x.shape[1], x.shape[2], x.shape[3]))
    x0        = xR[:, 0, ...]
    Tx0       = xR[:, 1, ...]
    
    # Reconstruction loss 
    loss_recon  = losses.l1_loss(x0, x0b_p) + losses.l1_loss(Tx0, Tx0b_p)
    
    # Total loss
    loss      = loss_recon + alpha_equiv * loss_equiv + beta_mult * loss_mult 
    
    return loss, (loss_recon, loss_equiv, loss_mult, z_p, x_out, tauOmegaBA, Omega)

  if train: 
    # Compute gradient
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, (loss_recon, loss_equiv, loss_mult, z, x_out, tauOmega, Omega)), grad = grad_fn(state.params)


    grad = jax.lax.pmean(grad, axis_name=PMAP_AXIS)
    grad = jax.tree_util.tree_map(jnp.conj, grad)

    updates, opt_state  = optimizer.update(grad, state.opt_state, state.params) 
    params              = optax.apply_updates(state.params, updates) 

    new_state           = state.replace( step=step,
                                         opt_state=opt_state,
                                         params=params,
                                         key=key)


    updater = TrainMetrics.gather_from_model_output

      
    
    metrics_update = updater(train_loss         = loss, 
                             train_recon_loss   = loss_recon,
                             train_equiv_loss   = loss_equiv, 
                             train_mult_loss    = loss_mult)
                             
  else:
    loss, (loss_recon, loss_equiv, loss_mult, z, x_out, tauOmega, Omega) = loss_fn(state.params) 


    metrics_update = EvalMetrics.single_from_model_output(eval_loss         = loss, 
                                                          eval_recon_loss   = loss_recon,
                                                          eval_equiv_loss   = loss_equiv,
                                                          eval_mult_loss    = loss_mult)
                                                         
    new_state = state.replace(key=key)
    
  return new_state, metrics_update, z, x_out, tauOmega, Omega
  
def tile_array( a ):
  C, m, n = a.shape

  h = int( np.ceil(np.sqrt(C)) )
  w = int( np.ceil(C/h) )

  out                       = np.zeros( (h, w, m, n), dtype=a.dtype )
  out.reshape(-1, m, n)[:C] = a
  out                       = out.swapaxes( 1, 2 ).reshape(-1, w * n)

  return out  


# Visualization w/ wandb
def viz_results( cfg, z, batch, x_out, tauOmega, Omega, mode="train", std_factor=2.5, image_dir=None):
  LH         = z.shape[1]
  LW         = z.shape[2]
  latent_dim = z.shape[3]
  
  H = x_out.shape[-3]
  W = x_out.shape[-2]

  assert jnp.isnan(z).any() == False 
  
  z                      = jnp.reshape( z, (-1, 4,  LH, LW, latent_dim) )
  [z0, gz0, z0_p, gz0_p] = jnp.split( z, 4, axis=1 )

  z, z_p = jnp.concatenate( (z0, gz0), axis=1), jnp.concatenate((z0_p, gz0_p), axis=1 )
  z      = jnp.reshape( z, (-1, LH, LW, latent_dim) )
  z_p    = jnp.reshape( z_p, (-1, LH, LW, latent_dim) )
  z      = jnp.concatenate( (z[:, None, ...], z_p[:, None, ...]), axis=1 )
  
  z_mean      = jnp.mean( z, axis=(2, 3), keepdims=True )
  z_std       = jnp.std( z, axis=(2, 3), keepdims=True )
  z_init_size = (z.shape[0], 2)

  z = (z - (z_mean - std_factor * z_std)) / ((z_mean + std_factor * z_std) - (z_mean - std_factor * z_std))
  z = jnp.clip( z, 0, 1 )

  z = jax.image.resize( z, z_init_size + (cfg.VIZ_SIZE[0] // 2, cfg.VIZ_SIZE[1] // 2, z.shape[-1]), method="nearest" )

  z_p = z[:, 1, ...]
  z   = z[:, 0, ...]

  z_p = jnp.reshape( z_p, (-1, 2, z.shape[1], z.shape[2], z.shape[3]))    
  z   = jnp.reshape( z, (-1, 2, z.shape[1], z.shape[2], z.shape[3]))


  batch = jax.image.resize(batch, (batch.shape[0], cfg.VIZ_SIZE[0], cfg.VIZ_SIZE[1], batch.shape[-1]), method="nearest")
      
  x_out = jax.image.resize(x_out, (x_out.shape[0], x_out.shape[1], cfg.VIZ_SIZE[0], cfg.VIZ_SIZE[1], x_out.shape[-1]), method="nearest")

  batchMax = jnp.max(batch, axis=(1, 2), keepdims=True)
  batchMin = jnp.min(batch, axis=(1, 2), keepdims=True)
  x_outMax = jnp.max(x_out, axis=(2, 3), keepdims=True)
  x_outMin = jnp.min(x_out, axis=(2, 3), keepdims=True)

  batch = (batch - batchMin) / ((batchMax - batchMin) + 1.0e-8)
  x_out = (x_out - x_outMin) / ((x_outMax - x_outMin) + 1.0e-8)

  
  batch = jnp.reshape(batch, (-1, 2, batch.shape[1], batch.shape[2], batch.shape[3]))
  assert jnp.isnan(batch).any() == False 

  x_out = jnp.transpose(x_out, (0, 1, 4, 2, 3))
  batch =jnp.transpose(batch, (0, 1, 4, 2, 3))


  x_outT = np.zeros((x_out.shape[0], x_out.shape[1]) + tile_array(x_out[0, 0, ...]).shape, dtype=x_out.dtype)
  batchT = np.zeros((batch.shape[0], batch.shape[1]) + tile_array(batch[0, 0, ...]).shape, dtype=batch.dtype)


  assert jnp.isnan(batch).any() == False
  for l in range( x_out.shape[0] ):
    for j in range( x_out.shape[1] ):
      x_outT[l, j, ...] = tile_array(x_out[l, j, ...])
    for j in range(batchT.shape[1]):
      batchT[l, j, ...] = tile_array(batch[l, j, ...])

  assert jnp.isnan(batch).any() == False 

  x_out = x_outT[..., None]
  batch = batchT[..., None]
  
  [x0, gx0, x0_p, gx0_p] = jnp.split(x_out, 4, axis=1) 

  x_out, x_out_p = jnp.concatenate((x0, gx0), axis=1), jnp.concatenate((x0_p, gx0_p), axis=1) 

  ic(batch.shape)
  ic(x_out.shape)

  [z, gz] = jnp.split(z, 2, axis=1)  
  [x, gx] = jnp.split(jnp.reshape(batch, (-1, 2, batch.shape[2], batch.shape[3], batch.shape[4])), 2, axis=1)
  [x0, gx0] = jnp.split(jnp.reshape(x_out, (-1, 2, batch.shape[2], batch.shape[3], batch.shape[4])), 2, axis=1)

  
  
  [z_p, gz_p] = jnp.split(z_p, 2, axis=1)
  [x0_p, gx0_p] = jnp.split(jnp.reshape(x_out_p, (-1, 2, batch.shape[2], batch.shape[3], batch.shape[4])), 2, axis=1)


  x_b = x.clone()
  gx_b = gx.clone()
  
  assert jnp.isnan(x).any() == False
  assert jnp.isnan(x0).any() == False
  assert jnp.isnan(x0_p).any() == False 
  
  x = np.asarray(255.0 * jnp.clip(jnp.concatenate((x, x0, x0_p), axis=2), 0, 1)).astype(np.uint8)
  gx = np.asarray(255.0 * jnp.clip(jnp.concatenate((gx, gx0, gx0_p), axis=2), 0, 1)).astype(np.uint8)


  z = np.transpose(np.asarray(255.0 * z).astype(np.uint8), (0, 1, 4, 2, 3))
  gz = np.transpose(np.asarray(255.0 * gz).astype(np.uint8), (0, 1, 4, 2, 3)) 


  z_p = np.transpose(np.asarray(255.0 * z_p).astype(np.uint8), (0, 1, 4, 2, 3))
  gz_p = np.transpose(np.asarray(255.0 * gz_p).astype(np.uint8), (0, 1, 4, 2, 3))
  
  
  zT = np.zeros( (z.shape[0], z.shape[1]) + tile_array(z[0, 0, ...]).shape, dtype=z.dtype)
  gzT = np.zeros_like(zT)


  zT_p = np.zeros_like(zT)
  gzT_p = np.zeros_like(zT)

  for l in range(gz.shape[0]):
    for j in range(gz.shape[1]):

      zT[l, j, ...] = tile_array(z[l, j, ...])
      gzT[l, j, ...] = tile_array(gz[l, j, ...])
      
      zT_p[l, j, ...] = tile_array(z_p[l, j, ...])
      gzT_p[l, j, ...] = tile_array(gz_p[l, j, ...])
  

  imB = np.concatenate((x, gx), axis=3)
  
  
  imL = np.concatenate((zT, jnp.zeros( (zT.shape[0], zT.shape[1], zT.shape[2], 10), dtype=np.uint8), gzT), axis=3)
  imL_p = np.concatenate((zT_p, jnp.zeros( (zT.shape[0], zT.shape[1], zT.shape[2], 10), dtype=np.uint8), gzT_p), axis=3)

  imL = np.concatenate((imL, jnp.zeros( (imL.shape[0], imL.shape[1], 10, imL.shape[-1]), dtype=np.uint8), imL_p), axis=2)


  
  tau_mean = jnp.mean(tauOmega, keepdims=True)
  tau_std = jnp.std(tauOmega, keepdims=True)
  tauR = (tauOmega - (tau_mean - std_factor * tau_std) ) / ((tau_mean + std_factor * tau_std) - (tau_mean - std_factor * tau_std))


  tauR = jax.image.resize(tauR, (tauR.shape[0], 1024, 1024), method='nearest')


  tauR = np.asarray(jnp.clip(tauR, 0, 1))

  tau_clr = np.reshape( 255.0 * np.asarray(clr_mapper_T.to_rgba(np.reshape(tauR, (-1, )).tolist()))[..., :3], tauR.shape + (3, )).astype(np.uint8 )


  Phi, Lambda = np.asarray( Omega[0]), np.asarray(Omega[1] )

  O      = np.matmul( Phi, Lambda[:, None] * np.swapaxes(Phi, -2, -1) )
  
  Phi_eigs = np.reshape( Phi, (LH, LW, Phi.shape[-1]) )  
  
  evecs_viz  = np.copy(Phi_eigs)
  evecs_mean = jnp.mean( evecs_viz )
  evecs_std  = jnp.std( evecs_viz )
  evecs_im   = (evecs_viz - (evecs_mean - 2.5 * evecs_std)) / ((evecs_mean + 2.5 * evecs_std) - (evecs_mean - 2.5 * evecs_std))
 

  evecs_im = np.clip( evecs_im, 0.0, 1.0 )
  evecs_im = np.asarray( jax.image.resize(jnp.asarray(evecs_im), (256, 256, evecs_im.shape[-1]), method='nearest') )
  
  evecs_imp = np.pad( evecs_im, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0 )
  
  evecs_imp = np.transpose( evecs_imp, (2, 0, 1) )
  evecs_imp = (255.0 * tile_array(evecs_imp)).astype(np.uint8) 
  evecs_imp = np.asarray( jax.image.resize(jnp.asarray(evecs_imp), (1024, 1024), method='nearest') )
  evecs_im = (255.0 * evecs_im).astype(np.uint8)
  
  O_mean = jnp.mean( O, keepdims=True )
  O_std  = jnp.std( O, keepdims=True )

  O = (O - (O_mean - std_factor * O_std) ) / ((O_mean + std_factor * O_std) - (O_mean - std_factor * O_std))
  O = np.asarray( jnp.clip(O, 0, 1) )
  
  O_clr = np.reshape( 255.0 * np.asarray(clr_mapper_K.to_rgba(np.reshape(O, (-1, )).tolist()))[..., :3], O.shape + (3, ) ).astype( np.uint8 )
  O_clr = np.asarray( jax.image.resize(jnp.asarray(O_clr), (1024, 1024, 3), method='nearest') )
 
  imBase_wandb   = []
  imLatent_wandb = []
  imTau_wandb    = []

  for l in range(x.shape[0]):
    imBase_wandb.append( wandb.Image(imB[l, ...], caption="{}_input_{}".format(mode, l)) )
    imLatent_wandb.append( wandb.Image(imL[l, ..., None], caption="{}_latents_{}".format(mode, l)) )
    imTau_wandb.append( wandb.Image(tau_clr[l, ...], caption="{}_tau_{}".format(mode, l)) )
  
  imO_wandb   = [wandb.Image(O_clr, caption="Operator")]
  evecs_wandb = [wandb.Image(evecs_imp, caption="Eigenfunctions")]

  wandb_ims = {
               "{}_reconstructions".format(mode): imBase_wandb,
               "{}_latent_features".format(mode): imLatent_wandb,
               "{}_tau".format(mode): imTau_wandb,
               "Eigenfunctions": evecs_wandb,
               "Learned Op.": imO_wandb
              }
               

  if image_dir is not None:
    learned_dir = os.path.join(image_dir, "learned")
  
    if not os.path.exists(learned_dir):
      os.makedirs(learned_dir)
  
    io.save_png(O_clr, os.path.join(learned_dir, "op.png"))
    io.save_png(evecs_imp, os.path.join(learned_dir, "eigs_all.png"))
  
    for k in range(tau_clr.shape[0]):
      io.save_png(tau_clr[k, ...], os.path.join(learned_dir, "tau_{}.png".format(k)))
  
    for l in range(evecs_im.shape[-1]):
      io.save_png(evecs_im[..., l], os.path.join(learned_dir, "eigs_{}.png".format(l)))
  
    samples_dir = os.path.join(image_dir, "samples")
  
    input_dir = os.path.join(samples_dir, "input")

  
    os.makedirs(input_dir, exist_ok=True)
  
    for k in range(xIm.shape[0]):
      
      io.save_png(np.asarray(xIm[k, ...]), os.path.join(input_dir, "x0_{}.png".format(k)))
      io.save_png(np.asarray(gxIm[k, ...]), os.path.join(input_dir, "gx0_{}.png".format(k)))
      
    np.savez(os.path.join(learned_dir, "raw.npz"), op=O, Phi=Phi_eigs, tau=tauOmega)
                   
               
  return wandb_ims 

 
def train_and_evaluate( cfg: Any, input_dir: str, output_dir: str ):
  tf.io.gfile.makedirs( output_dir )
  ic( output_dir )
  
  '''
  ========== Setup W&B =============
  '''
  project_name = cfg.PROJECT_NAME
  exp_name     =  "niso_{}_kopd_{}_ld_{}d_cshrec11_encode".format(cfg.KERNEL_OP_DIM, cfg.LATENT_DIM, len(cfg.CONV_ENC_CHANNELS)-1, cfg.DATASET_NAME)  

  cfg.WORK_DIR = output_dir 
  cfg.CSHREC11_DIR = input_dir
  image_dir = os.path.join(output_dir, "images")
  
  module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
  config         = module_to_dict( cfg )
 
  run = wandb.init( config=config, project=project_name, name=exp_name )
  
  '''
  ==================================
  '''


  ic( jax.default_backend() )

  ## Set up meta-parameters 
  true_batch_size = cfg.TRUE_BATCH_SIZE
  batch_size      = cfg.BATCH_SIZE 
  latent_dim      = cfg.LATENT_DIM 
  
  multi_steps  = true_batch_size // batch_size 
  alpha_equiv  = cfg.ALPHA_EQUIV
  beta_mult    = cfg.BETA_MULT 
  
  train_fn        = train_step
 
  viz_fn        = viz_results 
 
  input_size = cfg.INPUT_SIZE
  eval_every = cfg.EVAL_EVERY

  num_down = len(cfg.CONV_ENC_CHANNELS) - 1  
  latent_size = (input_size[0] // (2**num_down), input_size[1] // (2**num_down))
  
  ## Get dataset 
  print( "Getting dataset..." )
  train_data, test_data, NUM_CLASSES, stats = input_pipeline.get_cshrec11(cfg.CSHREC11_DIR)
  
  num_train_repeat = (cfg.NUM_TRAIN_STEPS * batch_size // cfg.DATA_TRAIN_SIZE) * 100
  num_test_repeat  = ((cfg.NUM_TRAIN_STEPS * batch_size * cfg.NUM_EVAL_STEPS * cfg.DATA_TEST_SIZE) // (eval_every ))*100 + 1 
  
  train_data = train_data.map( dl.zbound(stats), num_parallel_calls=tf.data.AUTOTUNE )
  train_data = train_data.shuffle( buffer_size=300, reshuffle_each_iteration=True )
  train_data = train_data.repeat( num_train_repeat )
  train_data = train_data.batch( batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE )

  test_data = test_data.map(dl.zbound(stats), num_parallel_calls=tf.data.AUTOTUNE)
  test_data = test_data.repeat( num_test_repeat )
  test_data = test_data.batch( batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE )

  train_iter = iter( train_data )
  test_iter  = iter( test_data )

  
  print( "Done..." )

  #Create models
  print( "Initializing models..." )
  encoder, decoder, kernel, optimizer, xform_key, state = create_train_state( cfg, (batch_size, *input_size, 16) )
  print( "Done..." )

  # Create checkpoints
  checkpoint_dir = os.path.join( output_dir, "checkpoints" )
  ckpt           = checkpoint.MultihostCheckpoint( checkpoint_dir, max_to_keep=2 )
  state          = ckpt.restore_or_initialize( state )
  
  initial_step = int(state.step) + 1

  # Distribute
  state = flax_utils.replicate( state )

  print( "Distributing..." )
  p_train_step = jax.pmap( functools.partial(train_fn,
                                             encoder      = encoder,
                                             decoder      = decoder,
                                             kernel       = kernel,
                                             optimizer    = optimizer,
                                             alpha_equiv  = alpha_equiv,
                                             beta_mult    = beta_mult,
                                             train        = True),
                           axis_name=PMAP_AXIS )
    
  p_eval_step = jax.pmap( functools.partial(train_fn,
                                            encoder      = encoder,
                                            decoder      = decoder,
                                            optimizer    = optimizer,
                                            kernel       = kernel,
                                            alpha_equiv  = alpha_equiv,
                                            beta_mult    = beta_mult,
                                            train        = False),
                          axis_name=PMAP_AXIS ) 
  # Visualize 
  train_metrics = None

  if cfg.STOP_AFTER is None:
    stop_at = cfg.NUM_TRAIN_STEPS + 1

  else:
    stop_at = cfg.STOP_AFTER + 1 
    
  print( "Beginning training..." )


  for step in tqdm( range(initial_step, stop_at) ):
    is_last_step = step == stop_at - 1

    ex = next( train_iter )
 
    batch = ex["image"]
    
    batch = jnp.asarray( batch.numpy() )
    
    assert jnp.isnan(batch).any() == False 

    xform_key, batch_key = jax.random.split( xform_key )

    batch = xforms.draw_shrec11_pairs(batch, xform_key)

    
    state, metrics_update, z, x_out, tauOmega, Omega = p_train_step( x=batch[None, ...], state=state )
    
    z     = z[0, ...]
    x_out = x_out[0, ...]
    tauOmega   = tauOmega[0, ...]
    Omega = (Omega[0][0, ...], Omega[1][0, ...], Omega[2][0, ...])
      
      
    assert np.isnan( np.asarray(z) ).any() == False 
      
    metric_update = flax_utils.unreplicate( metrics_update )
      
    train_metrics = (metric_update if train_metrics is None else train_metrics.merge(metric_update))

    # Visualize transformations if first step
    if step == initial_step:
      batchN = jnp.reshape( batch, (-1, 2, batch.shape[1], batch.shape[2], batch.shape[3]) )

      batchNMin = jnp.min(batchN, axis=(2, 3), keepdims=True)
      batchNMax = jnp.max(batchN, axis=(2, 3), keepdims=True)
      
      batchN = (batchN - batchNMin) / (batchNMax - batchNMin)
      
      im0, imX = np.split( np.asarray(batchN * 255.0).astype(np.uint8), 2, axis=1 )
      im0     = np.squeeze(im0[0, ...])
      imX     = np.squeeze(imX[0, ...])
      
      im0 = tile_array(jnp.transpose(im0, (2, 0, 1)))
      imX = tile_array(jnp.transpose(imX, (2, 0, 1)))

      im0 = np.concatenate( (im0, imX), axis=1)
      
      batch_im = [wandb.Image(im0[..., None], caption="batch_input_ex")]
      
      
      run.log({"batch_input_ex": batch_im})
      
    if step % cfg.VIZ_EVERY == 0 or is_last_step:
      wandb_ims = viz_fn( cfg, z, batch, x_out, tauOmega, Omega, mode="train" )

      run.log(wandb_ims, step=step)
      
    '''
    ===========================================
    ============== Eval Loop ==================
    ===========================================
    '''
      
    if step % cfg.EVAL_EVERY == 0 or is_last_step:
      eval_metrics = None 
      
      for j in range( cfg.NUM_EVAL_STEPS ):
      
        try:
          batch = next( test_iter )["image"]

        except:
          test_iter = iter( test_data )
          batch     = next( test_iter )["image"]
        
        batch = jnp.asarray( ex["image"].numpy() )

        xform_key, batch_key = jax.random.split( xform_key )

        batch = xforms.draw_shrec11_pairs(batch, xform_key)

        assert jnp.isnan(batch).any() == False 

        state, emetrics_update, z, x_out, tauOmega, Omega = p_eval_step( x=batch[None, ...], state=state )

        z     = z[0, ...]
        x_out = x_out[0, ...]
        tauOmega   = tauOmega[0, ...]
        Omega = (Omega[0][0, ...], Omega[1][0, ...], Omega[2][0, ...])

        emetric_update = flax_utils.unreplicate( emetrics_update )
          
        eval_metrics = (emetric_update if eval_metrics is None else eval_metrics.merge(emetric_update)) 

      wandb_ims = viz_fn( cfg, z, batch, x_out, tauOmega, Omega, mode="eval")
        
      run.log( data=eval_metrics.compute(), step=step )
      run.log( wandb_ims, step=step )

    if step % cfg.LOG_LOSS_EVERY == 0 or is_last_step:
      run.log( data=train_metrics.compute(), step=step )
      train_metrics = None
   
    if step % cfg.CHECKPOINT_EVERY == 0 or is_last_step:
      ckpt.save( flax_utils.unreplicate(state) )

  
'''
#######################################################
###################### Main ###########################
#######################################################
'''

if __name__ == '__main__':
  arguments = docopt( __doc__, version='Train 1.0' )

  # Set up experiment directory
  in_dir = arguments['--in']
  in_dir = in_dir.format( root_path=ROOT_PATH )

  out_dir = arguments['--out']
  out_dir = out_dir.format( root_path=ROOT_PATH )

  config = arguments['--config']
  path   = dirname( abspath(__file__) )

  spec = importlib.util.spec_from_file_location( "config", f"{path}/configs/{config}.py" )
  cfg  = importlib.util.module_from_spec(spec)

  sys.modules["module.name"] = cfg
  spec.loader.exec_module( cfg )

  if not os.path.exists( out_dir ):
    os.makedirs( out_dir )

  files = os.listdir( out_dir )
  count = 0
  
  for f in files:

    if os.path.isdir( os.path.join(out_dir, f) ) and f.isnumeric():
      count += 1

  exp_num = str( count )
  
  print( "==========================" )
  print( f"Experiment # {exp_num}" )
  print( "==========================" )
  
  exp_dir = os.path.join( out_dir, exp_num )
  os.mkdir( exp_dir )
  ic( exp_dir )
  
  train_and_evaluate( cfg, in_dir, exp_dir )

