
"""
Train.

Usage:
  train [options]
  train (-h | --help)
  train --version

Options:  
  -h --help                   Show this screen.
  --version                   Show version.
  -i, --in=<input_dir>        Input directory [default: {root_path}/data/CO3D/processed/].
  -w, --weights=<weight_dir>  Directory with pre-trained model weights from pose encode.
  -o, --out=<output_dir>      Output directory [default: {root_path}/experiments/pose_pred/weights/].
  -c, --config=<config>       Config directory [default: CO3D/config].

"""

import functools
import os
import sys
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
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import wandb
import scipy as sp 
import io

print(jax.default_backend(), flush=True)
tf.config.experimental.set_visible_devices( [], 'GPU' )

from flax.core.frozen_dict import freeze 
from docopt import docopt
from typing import Any, Callable, Dict, Sequence, Tuple, Union
from clu import checkpoint, metric_writers, metrics, parameter_overview, periodic_actions
from tqdm import tqdm
from icecream import ic

from os.path import dirname, abspath
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.append( ROOT_PATH )

from nn import losses, conv, fmaps, mlp
from utils import utils
from data import data_loader as dl 
from data import input_pipeline


# Constants
PMAP_AXIS = "batch" 

######################

@flax.struct.dataclass
class TrainState:
  step         : int
  opt_state    : Any
  params       : Any
  key          : Any 


def create_train_state( cfg: Any, data_shape: Tuple) -> Tuple[nn.Module, Any, Any, Any,  Any, TrainState]:

  # Random key 
  seed = 0 #np.random.randint(low=0, high=1e8, size=(1, ))[0]
  key                              = jax.random.PRNGKey( seed )
  key, enc_key, model_key, kernel_key, xform_key = jax.random.split( key, 5 )
  
  down_factor = np.power( 2, (len(cfg.CONV_ENC_CHANNELS) - 1) )

  ae_state = checkpoint.load_state_dict(cfg.PRE_TRAIN_DIR) 

  enc_params = ae_state["params"]["encoder"]
  kernel_params = ae_state["params"]["kernel"]

  encoder = conv.ConvEncoder( channels    = cfg.CONV_ENC_CHANNELS,
                              block_depth = cfg.CONV_ENC_BLOCK_DEPTH,
                              kernel_size = cfg.KERNEL_SIZE,
                              out_dim     = cfg.LATENT_DIM)

                                  
  dec_input    = jnp.ones( (data_shape[0], data_shape[1] // down_factor, data_shape[2] // down_factor, cfg.LATENT_DIM), dtype=jnp.float32 )
  kernel_input = jnp.reshape(dec_input, (dec_input.shape[0], -1,  cfg.LATENT_DIM))
            
  kernel = fmaps.operator_iso(op_dim=cfg.KERNEL_OP_DIM)
  
  # Initialize
  enc_input     = jnp.ones( data_shape, dtype=jnp.float32 )
  _             = encoder.init(enc_key, enc_input)
  _             = kernel.init(enc_key, kernel_input, kernel_input)
  
  # Pose predictor
  model = mlp.skipMLP(  features      = cfg.MLP_FEATURES,
                        num_layers    = cfg.MLP_LAYERS,
                        out_dim       = 9,
                        skip_init_act = True)
                                    

  xform_dim = cfg.KERNEL_OP_DIM

  model_input = jnp.ones( (data_shape[0], xform_dim * xform_dim), dtype=jnp.float32)

  model_params = model.init(model_key, model_input)["params"]

  params        = {"encoder": enc_params,  "kernel": kernel_params, "model": model_params} 


  # Set up optimizer 
  schedule = optax.warmup_cosine_decay_schedule( init_value   = cfg.INIT_LR,
                                                 peak_value   = cfg.LR,
                                                 warmup_steps = cfg.WARMUP_STEPS,
                                                 decay_steps  = cfg.NUM_TRAIN_STEPS,
                                                 end_value    = cfg.END_LR )

  batch_size = cfg.BATCH_SIZE
  num_multi_steps = cfg.TRUE_BATCH_SIZE // batch_size  
  
  optim = optax.adamw( learning_rate=schedule, b1=cfg.ADAM_B1, b2=cfg.ADAM_B2 )

  
  optimizer = optax.MultiSteps( optim, num_multi_steps )
  
  
  state       = optimizer.init( params ) 
  train_state = TrainState( step=0, opt_state=state, params=params, key=key )

  return model, encoder, kernel, optimizer, xform_key, train_state

@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  loss  : metrics.Average.from_output("loss")
  r_loss: metrics.Average.from_output("lossR")
  p_loss: metrics.Average.from_output("lossP")

  


@functools.partial(jax.jit, static_argnames=["max_step", "seq_len"]) 
def seq_sampler(image, g, mask, key, max_step, seq_len=4): 

 
 def _extract_seq(x, offsets):
  return x[offsets, ...] 
  
 step_key, choice_key, flip_key = jax.random.split(key, 3)
 
 c_keys = jax.random.split(choice_key, image.shape[0])
 
 
 steps = jax.random.randint(step_key, (image.shape[0], ), minval=1, maxval=max_step)
 
 steps = jnp.tile(steps[:, None], (1, seq_len-1))
 
 
 max_val = jnp.argmax(jnp.cumsum(mask, axis=-1)*mask, axis=-1)
 
 mask = 1.0 * jnp.greater( (max_val - (jnp.sum(steps, axis=-1) + 1))[:, None], jnp.cumsum(mask, axis=-1))
 
 index = jax.vmap(jax.random.choice, (0, None, None, None, 0, None), 0)(c_keys, mask.shape[-1], (), False, mask, 0)
 
 offsets = jnp.concatenate( (jnp.zeros((image.shape[0], 1), dtype=jnp.int32),  jnp.cumsum(steps, axis=-1)), axis=-1)
 
 x = jax.vmap(_extract_seq, (0, 0), 0)(image, offsets)
 
 gS = jax.vmap(_extract_seq, (0, 0), 0)(g, offsets)
 
 return jnp.reshape(x, (-1, image.shape[2], image.shape[3], image.shape[4])), gS
 

def train_step_seq(x: Any, g: Any, model: nn.Module, encoder: nn.Module, kernel: Any, state: TrainState,
                     optimizer: Any, seq_len: int):

  
  #key = state.key 
  step = state.step+1

  gRelF = jnp.zeros_like(g[:, 1:, ...])
  
  for l in range(seq_len-1):
    gRelF = gRelF.at[:, l, ...].set( jnp.matmul(utils.inv_se3(g[:, l, ...]), g[:, l+1, ...]))


  
  def loss_fn(params):
    
    z = encoder.apply({"params": params["encoder"]}, x)

 
    z        = jnp.reshape(z, (-1, seq_len, z.shape[1], z.shape[2], z.shape[3]))
    z        = jnp.reshape(z, (z.shape[0], z.shape[1], -1, z.shape[-1]))

  
      
    z0       = z[:, :-1, ...]
    Tz0      = z[:, 1:, ...]
    
    tauOmegaBA, _ = kernel.apply({"params": params["kernel"]}, 
                            jnp.reshape(z0, (-1, z0.shape[-2], z0.shape[-1])), 
                            jnp.reshape(Tz0, (-1, Tz0.shape[-2], Tz0.shape[-1])))

    
    tauOmegaBA = jax.lax.stop_gradient(tauOmegaBA)
    
    xformBA = jnp.reshape(tauOmegaBA, (tauOmegaBA.shape[0], -1))
    
    xiBA = model.apply({"params": params["model"]}, xformBA)

    gBA = utils.lift_9d_se3(jnp.reshape(xiBA, (-1, seq_len-1, xiBA.shape[-1])))
    
    lossR = losses.orientation_loss(gRelF[..., :3, :3], gBA[..., :3, :3])
        
    lossP = losses.procrustes_loss( gRelF[..., :3, -1], gBA[..., :3, -1] ) 

    loss  = lossR + lossP
    
    return loss, (lossR, lossP )

  # Compute gradient
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  (loss, (lossR, lossP)), grad = grad_fn(state.params)
 

  grad = jax.lax.pmean(grad, axis_name=PMAP_AXIS)
    
  grad = jax.tree_util.tree_map(jnp.conj, grad)

  updates, opt_state = optimizer.update(grad, state.opt_state, state.params) 
  params = optax.apply_updates(state.params, updates) 

  new_state = state.replace(
    step=step,
    opt_state=opt_state,
    params=params) 
    

  metrics_update = TrainMetrics.gather_from_model_output(loss=loss, lossR=lossR, lossP=lossP)

  return new_state, metrics_update

  

def estimate_rel_seq(seq: Any, g: Any, model: nn.Module, encoder: nn.Module, kernel: Any, state: TrainState):

  
  #key = state.key 
  step = state.step+1
  params = state.params 
  
  z = encoder.apply({"params": params["encoder"]}, seq)
  z = jnp.reshape(z, (z.shape[0], -1, z.shape[-1]))

 
  z0  = z[:-1, ...]
  Tz0 = z[1:, ...]
   
  tauOmegaBA, _ = kernel.apply({"params": params["kernel"]}, z0, Tz0)
                           
  xformBA = jnp.reshape(tauOmegaBA, (tauOmegaBA.shape[0], -1))
  
  xi = model.apply({"params": params["model"]}, xformBA)
    
  gRelP = utils.lift_9d_se3(xi)
  
  return gRelP

def tile_array( a ):
  C, m, n = a.shape

  h = int( np.ceil(np.sqrt(C)) )
  w = int( np.ceil(C/h) )

  out                       = np.zeros( (h, w, m, n), dtype=a.dtype )
  out.reshape(-1, m, n)[:C] = a
  out                       = out.swapaxes( 1, 2 ).reshape(-1, w * n)

  return out  



       
def train_and_evaluate( cfg: Any, input_dir: str, output_dir: str, load_weight_dir: str):
  tf.io.gfile.makedirs( output_dir )


  '''
  ========== Setup W&B =============
  '''
  exp_name     =  "niso_{}_kop_{}_ld_{}d_pose_pred".format( cfg.KERNEL_OP_DIM,
                                                            cfg.LATENT_DIM,
                                                            len(cfg.CONV_ENC_CHANNELS)-1)
  project_name      = cfg.PROJECT_NAME 
  cfg.WORK_DIR      = output_dir 
  cfg.PRE_TRAIN_DIR = load_weight_dir 
      
  module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
  config         = module_to_dict( cfg )

  run = wandb.init(config=config, project=project_name, name=exp_name)
  
  '''
  ==================================
  '''
  

  ## Set up meta-parameters 
  true_batch_size = cfg.TRUE_BATCH_SIZE
  batch_size      = cfg.BATCH_SIZE 
  latent_dim      = cfg.LATENT_DIM 
  
  num_multi_steps = true_batch_size // batch_size 
  
  max_step = cfg.TRAIN_MAX_STEP
  train_seq_len = cfg.TRAIN_SEQ_LEN
  
  train_fn        = train_step_seq
  sample_fn       = seq_sampler
  eval_fn         = estimate_rel_seq
  
  input_size = cfg.INPUT_SIZE
  eval_every = cfg.EVAL_EVERY

  num_down = len(cfg.CONV_ENC_CHANNELS) - 1  
  latent_size = (input_size[0] // (2**num_down), input_size[1] // (2**num_down))

  num_train_steps = cfg.NUM_TRAIN_STEPS * num_multi_steps
  
  ## Get dataset 
  print( "Getting dataset..." )

  train_data, test_data = input_pipeline.get_seq_dataset(input_dir)
  num_train_repeat = (num_train_steps * batch_size // cfg.DATA_TRAIN_SIZE) + 2

  
  train_data = train_data.map(dl.dual_images(), num_parallel_calls=tf.data.AUTOTUNE)
  #train_data = train_data.cache()
  train_data = train_data.shuffle(buffer_size = 500, reshuffle_each_iteration=True)
  train_data = train_data.repeat(num_train_repeat)
  train_data = train_data.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

  test_data = test_data.map(dl.dual_images(), num_parallel_calls=tf.data.AUTOTUNE)

  

  train_iter = iter(train_data)
  print( "Done..." )

  #Create models
  print( "Initializing models..." )    
    
  model, encoder, kernel, optimizer, xform_key, state = create_train_state(cfg, (batch_size, *input_size, 3))
  
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
                                             model        = model,
                                             encoder      = encoder,
                                             kernel       = kernel,
                                             optimizer    = optimizer,
                                             seq_len      = train_seq_len),
                           axis_name=PMAP_AXIS )
    
  p_eval_step = jax.pmap( functools.partial(eval_fn,
                                            model        = model,
                                            encoder      = encoder,
                                            kernel       = kernel),
                          axis_name=PMAP_AXIS )

  

  
  # Visualize 
  train_metrics = None

  if cfg.STOP_AFTER is None:
    stop_at = num_train_steps + 1

  else:
    stop_at = cfg.STOP_AFTER + 1 
    
  print( "Beginning training..." )


  for step in tqdm( range(initial_step, stop_at) ):
    is_last_step = step == stop_at - 1

    ex      = next(train_iter)
     
    batch   = ex["image"]
    gbatch  = ex["g"]
    mask    = ex["frame_mask"]
    
    batch   = jnp.asarray(batch.numpy())
    gbatch  = jnp.asarray(gbatch.numpy())
    mask    = jnp.asarray(mask.numpy())
    
    xform_key, batch_key = jax.random.split(xform_key)
    
    batch, gbatch        = sample_fn(batch, gbatch, mask, batch_key, max_step, train_seq_len)
    

    state, metrics_update = p_train_step(x=batch[None, ...], g=gbatch[None, ...], state=state)
    metric_update         = flax_utils.unreplicate(metrics_update)
      

    train_metrics = (metric_update if train_metrics is None else train_metrics.merge(metric_update))

      
    '''
    ===========================================
    ============== Eval Loop ==================
    ===========================================
    '''
   
            
    if step % cfg.EVAL_EVERY == 0 or is_last_step:
    
      traj_dir = os.path.join(output_dir, "traj")
      
      NUM_FAIL = []
      
      MAX_L = 200
      
      step_ate = {}
      for bstep in [1, 2, 4, 6, 8, 10]:
      
        traj_step_dir = os.path.join(traj_dir, "{}".format(bstep))
        
        if not os.path.exists(traj_step_dir):
          os.makedirs(traj_step_dir)
          
        fails = 0 


        min_seq_len = int(0.8 * jnp.arange(MAX_L, step=bstep).shape[0])

        sum_ate = 0.0 
        ate_count = 0 
                 
        for i, ex in enumerate(test_data):
            
          batch = jnp.asarray(ex["image"].numpy())
          gbatch = jnp.asarray(ex["g"].numpy())
          mask = jnp.asarray(ex["frame_mask"].numpy())
          qual = jnp.asarray(ex["quality"])[0]
          
          vID = ex["id"].numpy()[0].decode('UTF-8')
              
          seqL = jnp.sum(mask)
                 
          seq_ind = jnp.arange(min(MAX_L, seqL), step=bstep)
          
          if seq_ind.shape[0] < min_seq_len:
            continue
          
          try:
            gRelP = p_eval_step(seq=batch[None, seq_ind, ...], g=gbatch[None, seq_ind, ...], state=state)
          except:
            fails = fails + 1 
            continue 
              
          gRelP = np.asarray(gRelP[0, ...])

          
          #Recover predicted traj
          gTraj = np.asarray(gbatch[seq_ind, ...])
          
          gRel = np.zeros_like(gTraj[1:, ...])

          for l in range(len(seq_ind)-1):
            gRel[l, ...] = np.matmul(np.linalg.inv(gTraj[l, ...]), gTraj[l+1, ...])
            
          gTrajP = np.zeros_like(gTraj)
          gTrajP[0, ...] = gTraj[0, ...]
            
          for l in range(1, gTraj.shape[0]):   
            gTrajP[l, ...] = np.matmul(gTrajP[l-1, ...], gRelP[l-1, ...])
         
          
          pose_est = gTrajP[1:, :3, -1]
          pose_gt = gTraj[1:, :3, -1]
          
          
          try:
            pose_gt, pose_est, _ = sp.spatial.procrustes(pose_gt, pose_est)
          except:
            fails = fails + 1 
            continue

      
          eval_ate = np.mean((pose_est - pose_gt)**2)
          
          sum_ate = sum_ate + eval_ate 
          ate_count = ate_count + 1 
          
          
          poses = np.concatenate((gTraj[None, ...], gTrajP[None, ...]), axis=0)
          
          filename = os.path.join(traj_step_dir, vID + '.npy')
          np.save(filename, poses)
      
        mean_ate = np.sqrt(sum_ate / ate_count)
        
        run.log({"ate": mean_ate, "skip": bstep}) 
        
        NUM_FAIL.append(fails)

      ic(NUM_FAIL)
      run.log(step_ate)

  if step % cfg.LOG_LOSS_EVERY == 0 or is_last_step:
    run.log(data=train_metrics.compute(), step=step)
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

  weight_dir = arguments['--weights']
  
  config = arguments['--config']
  path   = dirname( abspath(__file__) )


  
  spec                       = importlib.util.spec_from_file_location( "config", f"{path}/configs/{config}.py" )
  cfg                        = importlib.util.module_from_spec(spec)
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

  
  train_and_evaluate( cfg, in_dir, exp_dir, weight_dir)

