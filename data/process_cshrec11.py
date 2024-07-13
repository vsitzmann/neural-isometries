import sys
import os
import gzip
import json

import jax 
import jax.numpy as jnp 

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np 
import scipy as sp
import numpy as np 

from collections import defaultdict
from tqdm import tqdm

from PIL import Image
from glob import glob 
from icecream import ic

from os.path import dirname, abspath
ROOT_PATH = dirname( dirname(abspath(__file__)) )
sys.path.append( ROOT_PATH )

from data import data_loader as dl 
from utils import ioutils as io
from utils import utils 

from data import xforms 

import trimesh as tmesh 

import scipy.sparse.linalg as sla
import potpourri3d as pp3d

NUM_EIGS = 128

S2_RES = (96, 192) 

NUM_CHANNELS = 16 

NUM_CLASS_TRAIN = 10 # 16
NUM_CLASS_TEST = 4 

DATA_DIR = '' # SET THIS TO: Directory storing unzipped Conformal Shrec 11 Dataset dataset (https://www.dropbox.com/scl/fi/nofmj3nfdzxm4uwhumo75/SHREC_11_CONF.zip?rlkey=3qst50619xg31bzax6jqzf3cm&st=r2qzzjbg&dl=0)

PROCESSED_DIR = '' # SET THIS TO: Directory storing output .tfrecord files

SKIP = ["glasses", "lamp", "snake", "two_balls", 'myScissor'] 

BASE_DIR = os.path.join(DATA_DIR, "base")
MOB_DIR = os.path.join(DATA_DIR, "mobius")
SBASE_DIR = os.path.join(DATA_DIR, "base_sphere")
SMOB_DIR = os.path.join(DATA_DIR, "mobius_sphere")

test_dir = os.path.join(PROCESSED_DIR, "test")
train_dir = os.path.join(PROCESSED_DIR, "train")

if not os.path.exists(test_dir):
  os.makedirs(test_dir)

if not os.path.exists(train_dir):
  os.makedirs(train_dir)

NUM_XFORMS = 30 

# Adapted from https://github.com/nmwsharp/diffusion-net
def get_spectrum(V, F, k=128):

  EPS = 1.0e-8
  eps = 1.0e-8 
  
  if F is not None:
    L = pp3d.cotan_laplacian(V, F, denom_eps=1e-10)
    mvec= pp3d.vertex_areas(V, F)
    mvec += eps * np.mean(mvec) 

  else:
    L, M = rl.point_cloud_laplacian(V) 
    mvec = M.diagonal() 
  

  L_eigsh = (L + sp.sparse.identity(L.shape[0])*EPS).tocsc()
  massvec_eigsh = mvec
  Mmat = sp.sparse.diags(massvec_eigsh)
  eigs_sigma = EPS
          
  # Prepare matrices
  failcount = 0
  while True:
    try:

      evals_np, evecs_np = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)

      # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
      evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

      break
    except Exception as e:
      print(e)
      if(failcount > 3):
          raise ValueError("failed to compute eigendecomp")
      failcount += 1
      print("--- decomp failed; adding eps ===> count: " + str(failcount))
      L_eigsh = L_eigsh + sp.sparse.identity(L.shape[0]) * (eps * 10**failcount)

  
  return evecs_np, evals_np, mvec 
  
'''
===========================================================================
============================ Main =========================================
===========================================================================
'''

if __name__ == '__main__':

  I, J = np.meshgrid(np.arange(S2_RES[0]), np.arange(S2_RES[1]), indexing="ij")
  
  I = np.reshape(I, (-1, ))
  J = np.reshape(J, (-1, ))
  
  I = (I / (S2_RES[0]-1)) * np.pi 
  
  J = (J / (S2_RES[1]-1)) * 2.0 * np.pi 
  
  X = np.cos(J) * np.sin(I)
  Y = np.sin(J) * np.sin(I)
  Z = np.cos(I)
  
  P = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
    
  
  classes = os.listdir(BASE_DIR)
  
  num_classes = len(classes)
  
  saved_keys = False 
  label_count = -1 

  MEAN = 0
  MEAN2 = 0 
  stat_count = 0 
  
  for l, c in enumerate(classes):

    
    if c in SKIP:
      continue


    label_count = label_count + 1 
  
    train_record = os.path.join(train_dir, "{}.tfrecords".format(c))
    test_record = os.path.join(test_dir, "{}.tfrecords".format(c))
    train_writer      = tf.io.TFRecordWriter( train_record )
    test_writer       = tf.io.TFRecordWriter( test_record ) 
    
    ic(c)

    CBASE_DIR  = os.path.join(BASE_DIR, c)
    CMOB_DIR   = os.path.join(MOB_DIR, c)
    CSBASE_DIR = os.path.join(SBASE_DIR, c)
    CSMOB_DIR  = os.path.join(SMOB_DIR, c)
    
    mesh_files = os.listdir(CBASE_DIR)

    mesh_files = mesh_files[np.random.permutation(len(mesh_files))][:NUM_CLASS_TRAIN+NUM_CLASS_TEST]
    
    deviance = []
    for j, mf in enumerate(mesh_files):

      mesh_ID = os.path.splitext(mf)[0]
      ic(mesh_ID) 
      # Load base mesh
      V, F = io.load_mesh(os.path.join(CBASE_DIR, mf))
      VS, _ = io.load_mesh(os.path.join(CSBASE_DIR, mf))
      Phi, Lambda, Mass = get_spectrum(V, F) 

      # Compute HKS 
      t = -1.0 * np.logspace(-2.0, 0.0, num=NUM_CHANNELS) 
      signal = np.sum( (Phi[..., None] ** 2) * np.exp( Lambda[None, :, None] * t[None, None, ...]), axis=-2)

      s_mean = np.sum(signal * Mass[:, None], axis=0) / np.sum(Mass)
      s_mean2 = np.sum( (signal ** 2) * Mass[:, None], axis=0) / np.sum(Mass)

      MEAN = MEAN + s_mean
      MEAN2 = MEAN2 + s_mean2 
      stat_count = stat_count + 1 
      
      mesh0 = tmesh.Trimesh(vertices=VS, faces=F)
      points0, _, tID0 = tmesh.proximity.closest_point(mesh0, P)
      bary0 = tmesh.triangles.points_to_barycentric(VS[F[tID0, :], :], points0, method='cross')

      base_signal = jnp.sum( signal[F[tID0, :], ...] * bary0[..., None], axis=1)        
      base_signal = np.reshape(base_signal, (S2_RES[0], S2_RES[1], NUM_CHANNELS))
      
      maps = np.zeros((NUM_XFORMS + 1, S2_RES[0], S2_RES[1], NUM_CHANNELS), dtype=np.float32)
      maps[0, ...] = base_signal 
      mob_xforms = np.zeros((NUM_XFORMS + 1, 2, 2, 2), dtype=np.float32)

      mob_xforms[0, :, :, 0] = np.eye(2)
      
      bad_indices = []
      for l in tqdm(range(NUM_XFORMS)):

        #try:
        VSM, _ = io.load_mesh(os.path.join(CSMOB_DIR, mesh_ID, "{}.ply".format(l)))
        meshM = tmesh.Trimesh(vertices=VSM,faces=F)
        pointsM, _, tIDM = tmesh.proximity.closest_point(meshM, P)
        assert np.isnan(pointsM).any() == False and np.isinf(pointsM).any() == False
        baryM = tmesh.triangles.points_to_barycentric(VSM[F[tIDM, :], :], pointsM, method='cross')
        assert np.isnan(baryM).any() == False and np.isinf(baryM).any() == False 
        mob_signal = jnp.sum( signal[F[tIDM, :], ...] * baryM[..., None], axis=1)
        mob_signal = np.reshape(mob_signal, (S2_RES[0], S2_RES[1], NUM_CHANNELS))
        assert np.isnan(mob_signal).any() == False and np.isinf(mob_signal).any() == False 

        maps[l+1, ...] = mob_signal 

        mob_xform, dM = xforms.estimate_mobius(VS, VSM)


        deviance.append(dM)

        mob_xforms[j, ..., 0] = np.real(mob_xform)
        mob_xforms[j, ..., 1] = np.imag(mob_xform)
        
        #except:
          #pass

      
      if bad_indices:
        ic(bad_indices)

        all_ind = np.arange(1, NUM_XFORMS+1).astype(np.int32)
        good_ind = np.setdiff1d(all_ind, np.asarray(bad_indices, dtype=np.int32))

        replace_ind = np.random.choice(good_ind, (len(bad_indices), ))

        for l in range(len(bad_indices)):
          maps[bad_indices[l], ...] = maps[replace_ind[l], ...]
  

      geom = {}
      geom["label"] = label_count * np.ones((NUM_XFORMS + 1, ), dtype=np.int32)
      geom["image"] = maps
      geom["xform"] = mob_xforms

      ex, shape_keys = dl.encode_example( geom )

      if not saved_keys:
        io.save_dict(os.path.join(test_dir, "shape_keys.json"), shape_keys)
        io.save_dict(os.path.join(train_dir, "shape_keys.json"), shape_keys)

      if j < NUM_CLASS_TRAIN:
        train_writer.write(ex)
      else:
        test_writer.write(ex)


    mean_dev = np.mean(deviance)
    std_dev = np.std(deviance)

    ic(mean_dev)
    ic(std_dev)
    
    test_writer.close()
    train_writer.close()
  
  MEAN = MEAN / stat_count 
  MEAN2 = MEAN2 / stat_count 

  ic(MEAN)
  ic(MEAN2)

  np.savez(os.path.join(PROCESSED_DIR, "stats.npz"), MEAN=MEAN, MEAN2=MEAN2)
      


  
    
  
 
