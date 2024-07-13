"""
Train.

Usage:
  train [options]
  train (-h | --help)
  train --version

Options:
  -h --help               Show this screen.
  --version               Show version.
  -o, --out=<output_dir>  Output directory [default: {root_path}/experiments/laplacian/weights/].
  -c, --config=<config>   Output directory [default: config].
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
import colormaps as cmaps
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
from data import input_pipeline
from utils import ioutils as io


# Constants
PMAP_AXIS = "batch" 

clr_norm   = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
clr_mapper_K = cm.ScalarMappable(norm=clr_norm, cmap=rwb_map)
clr_mapper_T = cm.ScalarMappable(norm=clr_norm, cmap=rwb_thick_map)
######################

@flax.struct.dataclass
class TrainState:
  step        : int
  opt_state   : Any
  params      : Any
  key         : Any 
  multi_steps : Any 

def resize(x, size):

  if x.ndim == 3:
    return np.asarray(jax.image.resize(jnp.asarray(x), (size[0], size[1], x.shape[-1]), method='nearest')) 
  else:   
    return np.asarray(jax.image.resize(jnp.asarray(x), (x.shape[0], size[0], size[1], x.shape[-1]), method='nearest'))
  
def create_train_state( cfg: Any) -> Tuple[nn.Module, Any, Any, TrainState]:

  # Random key 
  seed = 0 #np.random.randint(low=0, high=1e8, size=(1, ))[0]
  key                                          = jax.random.PRNGKey( seed )
  key, enc_key, dec_key, kernel_key, xform_key = jax.random.split( key, 5 )
  
  kernel = fmaps.operator_iso(op_dim=cfg.KERNEL_OP_DIM, clustered_init=True )
      
  kernel_input = jnp.ones((cfg.BATCH_SIZE, cfg.GRID_DIM[0] * cfg.GRID_DIM[1],  cfg.LATENT_DIM), dtype=jnp.float32)
            
  kernel_params = kernel.init( kernel_key, kernel_input, kernel_input )["params"]
  params        = {"kernel": kernel_params} 

  # Set up optimizer 
  schedule = optax.warmup_cosine_decay_schedule( init_value   = cfg.INIT_LR,
                                                 peak_value   = cfg.LR,
                                                 warmup_steps = cfg.WARMUP_STEPS,
                                                 decay_steps  = cfg.NUM_TRAIN_STEPS,
                                                 end_value    = cfg.END_LR )

  batch_size  = cfg.BATCH_SIZE
  multi_steps = cfg.TRUE_BATCH_SIZE // batch_size
  
  optim = optax.adamw( learning_rate=schedule, b1=cfg.ADAM_B1, b2=cfg.ADAM_B2 )
  
  optimizer = optax.chain( optax.clip(1.0), optim )
  optimizer = optax.MultiSteps( optim, multi_steps )
  
  state       = optimizer.init( params ) 
  train_state = TrainState( step=0, opt_state=state, params=params, key=key, multi_steps=multi_steps )
  
  return kernel, optimizer, xform_key, train_state


@flax.struct.dataclass
class TrainMetrics( metrics.Collection ):
  train_loss          : metrics.Average.from_output( "train_loss" )
  train_recon_loss    : metrics.Average.from_output( "train_recon_loss" )
  train_mult_loss     : metrics.Average.from_output( "train_mult_loss")


@functools.partial(jax.jit, static_argnames=["B"]) 
def get_shift(x, key, B):
  
  def _shift(a, t_x, t_y):
    return jnp.roll(a, (t_x, t_y), axis=(0, 1))
  
  
  H, W, C = x.shape[1], x.shape[2], x.shape[3]
  
  x = jnp.reshape(x, (B, -1, H, W, C))
  x = jnp.transpose(x, (0, 2, 3, 1, 4))
  x = jnp.reshape(x, (B, H, W, -1))
  
  
  x_key, y_key = jax.random.split(key, 2)
  
  t_x = jax.random.randint(x_key, shape=(B, ), minval=0, maxval=H)
  t_y = jax.random.randint(y_key, shape=(B, ), minval=0, maxval=W)
  
  Tx = jax.vmap(_shift, (0, 0, 0), 0)(x, t_x, t_y)
  
  return x, Tx 
  


  
def train_step(x: Any, Tx: Any, kernel: nn.Module,  state: TrainState, optimizer: Any, beta_mult: float):
  
  step = state.step+1
  H, W = x.shape[1], x.shape[2]
  key = state.key 
  
  
  def loss_fn(params):

    z0 = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))
    Tz0 = jnp.reshape(Tx, (x.shape[0], -1, Tx.shape[-1]))
    
    tauOmegaBA, Omega  = kernel.apply({"params": params["kernel"]}, z0, Tz0)
    tauOmegaAB         = jnp.swapaxes(tauOmegaBA, -2, -1)  
          
    Tz0_p = jnp.einsum( "...ij, ...jk, ...lk, ...lm->...im", Omega[0][None, ...], tauOmegaBA, Omega[0][None, ...], Omega[2][None, ..., None] * z0 )
    z0_p  = jnp.einsum( "...ij, ...jk, ...lk, ...lm->...im", Omega[0][None, ...], tauOmegaAB, Omega[0][None, ...], Omega[2][None, ..., None] * Tz0 )

    loss_mult  = losses.multiplicity_loss(Omega[1])

    loss_recon = losses.l1_loss(z0, z0_p) + losses.l1_loss(Tz0, Tz0_p)
    
    loss       = loss_recon + beta_mult * loss_mult 

    x_p  = jnp.reshape(z0_p, (x.shape[0], H, W, x.shape[-1]))
    Tx_p = jnp.reshape(Tz0_p, (x.shape[0], H, W, x.shape[-1]))
    
    return loss, (loss_recon, loss_mult, x_p, Tx_p, tauOmegaBA, Omega)


  # Compute gradient
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  (loss, (loss_recon, loss_mult, x_p, Tx_p, tauOmega, Omega)), grad = grad_fn(state.params)

  grad = jax.lax.pmean(grad, axis_name=PMAP_AXIS)
  grad = jax.tree_util.tree_map(jnp.conj, grad)

  updates, opt_state = optimizer.update(grad, state.opt_state, state.params) 
  params = optax.apply_updates(state.params, updates) 

  new_state = state.replace(
    step=step,
    opt_state=opt_state,
    params=params,
    key=key)

  metrics_update = TrainMetrics.gather_from_model_output(train_loss         =loss,
                                                           train_recon_loss =loss_recon,
                                                           train_mult_loss  =loss_mult)

    
  return new_state, metrics_update, x_p, Tx_p, tauOmega, Omega
  
def tile_array( a ):
  C, m, n = a.shape

  h = int( np.ceil(np.sqrt(C)) )
  w = int( np.ceil(C/h) )

  out                       = np.zeros( (h, w, m, n), dtype=a.dtype )
  out.reshape(-1, m, n)[:C] = a
  out                       = out.swapaxes( 1, 2 ).reshape(-1, w * n)

  return out  


def viz_results(x, Tx, x_p, Tx_p, tauOmega, Omega, image_dir=None, std_factor=2.5):

  mode = "train"
  
  num_ims = 9
 
  H, W = x.shape[1], x.shape[2]
  
  x = np.asarray(x)
  Tx = np.asarray(Tx)
  x_p = np.asarray(x_p)
  Tx_p = np.asarray(Tx_p)
  
  x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], -1, 3))
  Tx = np.reshape(Tx, (Tx.shape[0], Tx.shape[1], Tx.shape[2], -1, 3))
  x_p = np.reshape(x_p, (x_p.shape[0], x_p.shape[1], x_p.shape[2], -1, 3)) 
  Tx_p = np.reshape(Tx_p, (Tx_p.shape[0], Tx_p.shape[1], Tx_p.shape[2], -1, 3))
  
  x = np.transpose(x, (0, 3, 1, 2, 4))
  Tx = np.transpose(Tx, (0, 3, 1, 2, 4))
  x_p = np.transpose(x_p, (0, 3, 1, 2, 4))
  Tx_p = np.transpose(Tx_p, (0, 3, 1, 2, 4))
  
  x = np.reshape(x, (-1, H, W, 3))
  Tx = np.reshape(Tx, (-1, H, W, 3))
  x_p = np.reshape(x_p, (-1, H, W, 3))
  Tx_p = np.reshape(Tx_p, (-1, H, W, 3))
  
  x = (255 * 0.5 * (x + 1.0)).astype(np.uint8)
  Tx = (255 * 0.5 * (Tx + 1.0)).astype(np.uint8)  
  x_p = (255 * 0.5 * (x_p + 1.0)).astype(np.uint8) 
  Tx_p = (255 * 0.5 * (Tx_p + 1.0)).astype(np.uint8)
  
  x = x[:num_ims, ...]
  Tx = Tx[:num_ims, ...]
  x_p = x_p[:num_ims, ...]
  Tx_p = Tx_p[:num_ims, ...]
  
  
  x = np.asarray(jax.image.resize(x, (x.shape[0], 64, 64, 3), method="nearest"))
  Tx = np.asarray(jax.image.resize(Tx, (Tx.shape[0], 64, 64, 3), method="nearest"))  
  x_p = np.asarray(jax.image.resize(x_p, (x_p.shape[0], 64, 64, 3), method="nearest"))  
  Tx_p = np.asarray(jax.image.resize(Tx_p, (Tx_p.shape[0], 64, 64, 3), method="nearest"))
  
  xT = np.zeros( tile_array(x[..., 0]).shape + (3, ), dtype=np.uint8)
  TxT = np.zeros_like(xT)
  
  xpT = np.zeros_like(xT)
  TxpT = np.zeros_like(xT)


  for l in range(3):
    xT[..., l] = tile_array(x[..., l])
    TxT[..., l] = tile_array(Tx[..., l])    
    xpT[..., l] = tile_array(x_p[..., l])
    TxpT[..., l] = tile_array(Tx_p[..., l])
    

  ims = np.concatenate((xT, np.zeros( (xT.shape[0], 10, xT.shape[2]), dtype=np.uint8), TxT), axis=1)
  ims_p = np.concatenate((xpT, np.zeros( (xT.shape[0], 10, xT.shape[2]), dtype=np.uint8), TxpT), axis=1)
  
  ims = np.concatenate((ims, np.zeros((10, ims.shape[1], ims.shape[2]), dtype=np.uint8), ims_p), axis=0)
  tauR = (tauOmega[:, None, ...] + 1.0) / 2.0 
  
  tauR = jax.image.resize( tauR, (tauR.shape[0], tauR.shape[1], 1024, 1024), method='nearest' )
  tauR = np.asarray( jnp.clip(tauR, 0, 1) )

  tau_clr = np.reshape( 255.0 * np.asarray(clr_mapper_T.to_rgba(np.reshape(tauR, (-1, )).tolist()))[..., :3], tauR.shape + (3, ) ).astype( np.uint8 )

  tauT = np.zeros( (tauR.shape[0], ) + tile_array(tau_clr[0, ..., 0]).shape + (3, ), dtype=tau_clr.dtype )

  for l in range( tau_clr.shape[0] ):
    for j in range( 3 ):
      tauT[l, ..., j] = tile_array( tau_clr[l, ..., j] )

  Phi, Lambda = np.asarray( Omega[0]), np.asarray(Omega[1] )

  O      = np.matmul( Phi, Lambda[:, None] * np.swapaxes(Phi, -2, -1) )
  
  Phi_eigs = np.reshape( Phi, (H, W, Phi.shape[-1]) )  
  
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
 
  imO_wandb = [wandb.Image(O_clr, caption="Learned_Op")]
  evecs_wandb = [wandb.Image(evecs_imp, caption="Learned_Eigs")]
  ims_wandb = [wandb.Image(ims, caption="Images")]

  imTau_wandb    = []

  for l in range(tauR.shape[0]):
    imTau_wandb.append( wandb.Image(tauT[l, ...], caption="{}_tau_{}".format(mode, l)) )
    
  wandb_ims = {"Images": ims_wandb,
               "Learned_Op": imO_wandb,
               "Leanred_Eigs": evecs_wandb,
               "tauOmega": imTau_wandb}
               

  if image_dir is not None:
    learned_dir = os.path.join(image_dir, "learned")
  
    if not os.path.exists(learned_dir):
      os.makedirs(learned_dir)
  
    io.save_png(O_clr, os.path.join(learned_dir, "op.png"))
    io.save_png(evecs_imp, os.path.join(learned_dir, "eigs_all.png"))
  
    for k in range(tauT.shape[-1]):
      io.save_png(tauT[l, ...], os.path.join(learned_dir, "tau_{}.png".format(k)))
  
    for l in range(evecs_im.shape[-1]):
      io.save_png(evecs_im[..., l], os.path.join(learned_dir, "eigs_{}.png".format(l)))
  
    np.savez(os.path.join(learned_dir, "raw.npz"), op=O, Phi=K_eigs, tau=tauOmega)
  
  return wandb_ims
               



 
def train_and_evaluate( cfg: Any, output_dir: str ):
  tf.io.gfile.makedirs( output_dir )

  
  '''
  ========== Setup W&B =============
  '''
  project_name = cfg.PROJECT_NAME
  exp_name     =  "niso_toric_laplacian"

  cfg.WORK_DIR = output_dir 

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
  
  beta_mult = cfg.BETA_MULT

  grid_dim = cfg.GRID_DIM
  

  train_fn        = train_step
  

  viz_fn        = viz_results
  sample_fn     = get_shift 


  num_train_steps = cfg.NUM_TRAIN_STEPS
  
  Phi, Lambda, M, Lap = utils.get_toric_eigs(*grid_dim, cfg.KERNEL_OP_DIM)

  ## Get dataset 
  print("Getting dataset...", flush=True)

  train_data = tfds.load('imagenet_resized/{}x{}'.format(cfg.GRID_DIM[0], cfg.GRID_DIM[0]), split='train', shuffle_files=True)
  num_ims = 1281167
  
  ims_per_batch = (latent_dim // 3) + 1
  
  im_in_batch = batch_size * ims_per_batch 
  
  num_train_repeat = (num_train_steps * im_in_batch // num_ims) + 2 
  
  train_data = train_data.map(dl.dual_images(), num_parallel_calls=tf.data.AUTOTUNE)
  train_data = train_data.cache()
  train_data = train_data.shuffle(buffer_size= 100 * im_in_batch, reshuffle_each_iteration=True)
  train_data = train_data.repeat(num_train_repeat)
  train_data = train_data.batch(im_in_batch, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
  
  train_iter = iter(train_data)

  
  print("Done...", flush=True)

  #Create models
  print("Initializing models...", flush=True)    
  kernel, optimizer, xform_key, state = create_train_state(cfg)
  print("Done...", flush=True)

  # Create checkpoints
  checkpoint_dir = os.path.join( output_dir, "checkpoints" )
  ckpt           = checkpoint.MultihostCheckpoint( checkpoint_dir, max_to_keep=2 )
  state          = ckpt.restore_or_initialize( state )
  
  initial_step = int(state.step) + 1

  # Distribute
  state = flax_utils.replicate( state )

  print( "Distributing..." )
  p_train_step = jax.pmap( functools.partial(train_fn,
                                             kernel       = kernel,
                                             optimizer    = optimizer,
                                             beta_mult    = beta_mult),
                          axis_name=PMAP_AXIS)

    
  # Visualize 
  train_metrics = None
  print("Beginning training...", flush=True)
  if cfg.STOP_AFTER is None:
    stop_at = num_train_steps+1
  else:
    stop_at = cfg.STOP_AFTER + 1 

  
  #with metric_writers.ensure_flushes(writer):
  for step in tqdm(range(initial_step, stop_at)):
    is_last_step = step == stop_at-1

 
    ex = next(train_iter)
    
    batch = ex["image"]

    batch = jnp.asarray(batch.numpy())

    xform_key, batch_key = jax.random.split(xform_key)


    batch, Tbatch = sample_fn(batch, batch_key, batch_size)
    
    state, metrics_update, x_out, Tx_out, tauOmega, Omega = p_train_step(x=batch[None, ...], Tx=Tbatch[None, ...], state=state)
          
    x_out = x_out[0, ...]
    Tx_out = Tx_out[0, ...]
    
    tauOmega = tauOmega[0, ...]
    Omega = (Omega[0][0, ...], Omega[1][0, ...], Omega[2][0, ...])
    
    metric_update = flax_utils.unreplicate(metrics_update)
     
    train_metrics = (
          metric_update
          if train_metrics is None else train_metrics.merge(metric_update))

    
    if step == initial_step:
      std_factor = 2.5
      Phi = np.reshape(Phi, (grid_dim[0], grid_dim[1], -1))

      evecs_viz = Phi
      evecs_mean = jnp.mean(evecs_viz)
      evecs_std = jnp.std(evecs_viz) 
      evecs_im = (evecs_viz - (evecs_mean - 2.5 * evecs_std)) / ((evecs_mean + 2.5 * evecs_std) - (evecs_mean - 2.5 * evecs_std))
      
      evecs_im = np.clip(evecs_im, 0.0, 1.0)
      evecs_im = np.asarray(jax.image.resize(jnp.asarray(evecs_im), (256, 256, evecs_im.shape[-1]), method='nearest'))
      
      evecs_imp = np.pad(evecs_im, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)
      
      
      evecs_imp = np.transpose(evecs_imp, (2, 0, 1))
      
      evecs_imp = (255.0 * tile_array(evecs_imp)).astype(np.uint8) 
      evecs_im = (255.0 * evecs_im).astype(np.uint8) 
            
      K_mean = jnp.mean(Lap, keepdims=True)
      K_std = jnp.std(Lap, keepdims=True)
      K = (Lap - (K_mean - std_factor * K_std) ) / ((K_mean + std_factor * K_std) - (K_mean - std_factor * K_std))
      
      K = np.asarray(jnp.clip(K, 0, 1))

      K_clr = np.reshape(255.0 * np.asarray(clr_mapper_K.to_rgba(np.reshape(K, (-1, )).tolist()))[..., :3], K.shape + (3, )).astype(np.uint8) 
      
      K_clr = np.asarray(jax.image.resize(jnp.asarray(K_clr), (1024, 1024, 3), method='nearest'))
        
      imK_wandb = [wandb.Image(K_clr, caption="Laplacian")]
      evecs_wandb = [wandb.Image(evecs_imp, caption="Lap Eigs")]


      wandb_ims = {"Laplacian": imK_wandb,
                   "Lap Eigs": evecs_wandb}

      lap_dir = os.path.join(image_dir, "lap")

      '''
      if not os.path.exists(lap_dir):
        os.makedirs(lap_dir)

      io.save_png(K_clr, os.path.join(lap_dir, 'lap.png'))
      io.save_png(evecs_imp, os.path.join(lap_dir, 'eig_tile.png'))

      for l in range(evecs_im.shape[-1]):
        io.save_png(evecs_im[..., l], os.path.join(lap_dir, "eig_{}.png".format(l)))

      np.savez(os.path.join(lap_dir, "raw.npz"), Phi=Phi, Lap=Lap)
      '''
                   
      run.log(wandb_ims, step=step)
      
    if step % cfg.VIZ_EVERY== 0 or is_last_step:
      
      wandb_ims = viz_fn(batch, Tbatch, x_out, Tx_out, tauOmega, Omega) 
      
      run.log(wandb_ims, step=step)
      
    if step % cfg.LOG_LOSS_EVERY == 0 or is_last_step:
      run.log(data=train_metrics.compute(), step=step)
      train_metrics = None

    
    if step % cfg.CHECKPOINT_EVERY == 0 or is_last_step:
      ckpt.save(flax_utils.unreplicate(state))
        
'''
#######################################################
###################### Main ###########################
#######################################################
'''

if __name__ == '__main__':
  arguments = docopt( __doc__, version='Train 1.0' )

  # Set up experiment directory
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
  
  train_and_evaluate( cfg, exp_dir )