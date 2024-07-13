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

med_res = (224, 224)
low_res = (144, 144) 

from PIL import Image
from glob import glob 
from icecream import ic

from os.path import dirname, abspath
ROOT_PATH = dirname( dirname(abspath(__file__)) )
sys.path.append( ROOT_PATH )

from data import data_loader as dl 
from utils import ioutils as io
from utils import utils 


DATA_DIR = "" # SET THIS TO: Directory storing unzipped CO3Dv2 dataset
PROCESSED_DIR = "" # SET THIS TO: Directory storing output .tfrecord files


SEQ_NAMES = os.listdir(DATA_DIR)

SPLITS = [("train", 10), ("test", 1)] 

TEST_PERCENT = 0.1 
#SPLITS = [("test", 1)]

'''
===========================================================================
============================ Helper =======================================
===========================================================================
'''

def square_crop(image):

    H, W = image.shape[0], image.shape[1] 
    
    N = min(H, W) 
    
    dH = (H - N) // 2
    dW = (W - N) // 2
    
    h0 = dH
    h1 = dH + N 
    w0 = dW
    w1 = dW + N 
    
    return image[h0:h1, w0:w1, :]
    

def square_crop_batch(image):

    H, W = image.shape[1], image.shape[2] 
    
    N = min(H, W) 
    
    dH = (H - N) // 2
    dW = (W - N) // 2
    
    h0 = dH
    h1 = dH + N 
    w0 = dW
    w1 = dW + N 
    
    return image[:, h0:h1, w0:w1, :]


    
def _load_16big_png_depth( depth_png ) -> np.ndarray:

    with Image.open( depth_png ) as depth_pil:
        # The image is stored with 16-bit depth but PIL reads it as I (32 bit).  We cast it to uint16, then reinterpret as float16, then cast to float32
        depth = ( np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                  .astype(np.float32)
                  .reshape((depth_pil.size[1], depth_pil.size[0])) )

    return depth


def _load_depth( path, scale_adjustment ) -> np.ndarray:
    d                  = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0

    return d[None]  # fake feature channel


def count_frames():

  valid_splits = io.load_dict("co3dv2_top25_splits.json")
  
  MF = {}
  for (s, num_shards) in SPLITS:
    

      
    max_frames = 0
    for seq_name in tqdm(SEQ_NAMES):
    
      seq_dir = os.path.join( DATA_DIR, seq_name )
      frame_data = json.loads( gzip.GzipFile(os.path.join(seq_dir, "frame_annotations.jgz"), "rb").read().decode("utf8") ) # Parse dataset  
      
      valid_splits = valid_splits[s][seq_name]
      
      sequences = defaultdict( list )
      
      for i, data in enumerate( frame_data ):
      
        if data["sequence_name"] in valid_seqs:
        
          sequences[data["sequence_name"]].append( data )

      
      seqs = {}
      for i, (k, v) in enumerate( sequences.items() ):

          
        seqs[k] = sorted( sequences[k], key=lambda x:x["frame_number"] )
      
      
      for (k, v) in seqs.items():
        n_frames = len( v )
        
        if n_frames > max_frames:
          max_frames = n_frames
      
    
    MF[s] = max_frames
    

  return MF
  
       

'''
===========================================================================
============================ Main =========================================
===========================================================================
'''

if __name__ == '__main__':

  
  MF = count_frames()
  
  exit()
  assert False == True 
  
  valid_splits = io.load_dict("co3dv2_top25_splits.json")
     
  
  for (s, num_shards) in SPLITS:  
    max_frames = MF[s]
    ic(max_frames)
    split_count = 0 
    for seq_name in SEQ_NAMES:
      seq_dir = os.path.join( DATA_DIR, seq_name )
    
      valid_seqs = valid_splits[s][seq_name]
            
      frame_data = json.loads( gzip.GzipFile(os.path.join(seq_dir, "frame_annotations.jgz"), "rb").read().decode("utf8") ) # Parse dataset

      seq_data = json.loads( gzip.GzipFile(os.path.join(seq_dir, "sequence_annotations.jgz"), "rb").read().decode("utf8") )

      
      sequences = defaultdict( list )
      sdat = defaultdict(list)
            
      for i, data in enumerate( frame_data ):
        
        if data["sequence_name"] in valid_seqs:
        
          sequences[data["sequence_name"]].append( data )
      
            
      for i, data in enumerate(seq_data):
        if data["sequence_name"] in valid_seqs:
          sdat[data["sequence_name"]].append(data)
      
        
        
      seqs = {}
      for i, (k, v) in enumerate( sequences.items() ):
          
        seqs[k] = sorted( sequences[k], key=lambda x:x["frame_number"] )
      
          
      # Shard it up 
      save_dir = os.path.join( PROCESSED_DIR, s )
      os.makedirs( save_dir, exist_ok=True )

      k_splits = np.array_split( list(seqs.keys()), num_shards )
      #ic( k_splits )
      
      saved_keys = False 
      
      pbar = tqdm(total=len(seqs.keys()))
      
      for l in range( len(k_splits) ):
        record_name = save_dir + "/seqs_{}.tfrecords".format(split_count)
        writer      = tf.io.TFRecordWriter( record_name )
        split_count = split_count + 1 
        
        for k in k_splits[l]: #tqdm( k_splits[l] ):
          
          # Example quantities
          images     = np.zeros( (max_frames, low_res[0], low_res[1], 3), dtype=np.uint8 )
          #images_med = np.zeros( (max_frames, med_res[0], med_res[1], 3), dtype=np.uint8 )
          #depth      = np.zeros( (max_frames, low_res[0], low_res[1], 1), dtype=np.float32 )
          #depth_med  = np.zeros( (max_frames, med_res[0], med_res[1],  1), dtype=np.float32 )
          g          = np.tile( np.eye(4)[None, ...], (max_frames, 1, 1) )
          mask       = np.zeros( max_frames, dtype=np.int32 )
          quality    = np.zeros( (1,), dtype=np.float32)
          
          traj_id = seq_name + '_' + k
          
          num_seq_frames = len(seqs[k])
          
          mask[:num_seq_frames] = np.ones(num_seq_frames, dtype=np.int32)
          
          pose_quality = sdat[k][0]["viewpoint_quality_score"]
          
          quality[0] = pose_quality
          
          #ic(pose_quality)
          
          RMat = []
          tMat = []
          imMat = []
          

          
          for i in range( num_seq_frames ):

             
            im_path    = seqs[k][i]["image"]["path"]
            depth_path = seqs[k][i]["depth"]["path"]
            scale_adj  = seqs[k][i]["depth"]["scale_adjustment"]
            
            R = np.asarray(seqs[k][i]["viewpoint"]["R"]).T
            t = np.asarray(seqs[k][i]["viewpoint"]["T"])
            
            RMat.append(R[None, ...])
            tMat.append(t[None, ...])
            
            # Load image
            im = io.load_png(os.path.join(DATA_DIR, im_path)) 
            
            imMat.append(im[None, ...])
         
          imMat = np.concatenate(imMat, axis=0)
          RMat  = np.concatenate(RMat, axis=0)
          tMat  = np.concatenate(tMat, axis=0)
          
          imMat = square_crop_batch(imMat)
          imMat = np.asarray(jax.image.resize(jnp.asarray(imMat), (imMat.shape[0], low_res[0], low_res[1], 3), method='lanczos3')).astype(np.uint8)
          
          images[:num_seq_frames, ...] = imMat 
          
          pose = np.concatenate((RMat, tMat[..., None]), axis=-1)
          pose = np.matmul(np.diag( [-1, -1, 1] ).astype(np.float32)[None, ...], pose)
          
          g[:num_seq_frames, :3, :4] = pose 
          
          g = np.linalg.inv(g) 
          
          
 
          vid_dict = {}
          
          vid_dict["image"]      = images
          #vid_dict["image_med"]  = images_med
          #vid_dict["depth"]      = depth
          #vid_dict["depth_med"]  = depth_med
          vid_dict["g"]          = g 
          vid_dict["frame_mask"] = mask 
          vid_dict["quality"] = quality 
          vid_dict["id"] = traj_id 
          
          ex, shape_keys = dl.encode_example( vid_dict )
          
          if not saved_keys:
            dict_file = os.path.join( save_dir, "shape_keys.json" )
            #print(dict_file, flush=True)
            #print(shape_keys, flush=True)
            
            io.save_dict( dict_file, shape_keys )
            saved_keys = True
            
          writer.write(ex)
          
          pbar.update(1)
        
        writer.close() 
      
      pbar.close()
