import tensorflow as tf
import tensorflow_datasets as tfds
import jax 
import jax.numpy as jnp 
import numpy as np 
import random 

import glob
import json
import gzip 

from PIL import Image

Glob = glob.glob


def load_dict(file_name):
  
  with open( file_name ) as json_file:
    data = json.load( json_file )
  
  return data 


def encode_example(geom):
  features = {}
  
  key_list = {}
  for k in geom:
    
    feat = geom[k]
    
    
    if isinstance(feat, str):
      feat = tf.convert_to_tensor([feat], dtype=tf.string)
      key_list[k] = ["string", (1, )]
      
    elif (feat.dtype.char in np.typecodes["AllFloat"]):
      feat = tf.convert_to_tensor(feat.astype(np.float32), dtype=tf.float32)
      key_list[k] = ["float", geom[k].shape]
    
    elif (feat.dtype == "uint8"):
      feat = tf.convert_to_tensor(feat.astype(np.uint8), dtype=tf.uint8)
      key_list[k] = ["uint8", geom[k].shape]
      
    else:
      feat = tf.convert_to_tensor(feat.astype(np.int32), dtype=tf.int32)
      key_list[k] = ["int", geom[k].shape]
      
    
    feat_serial = tf.io.serialize_tensor(feat)

    features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat_serial.numpy()]))
      
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString(), key_list

      
def encode_shape_example(geom):
  
  features = {}
  
  key_list = {}
  for k in geom:
    
    
    if (k == "num_levels"):
      features["num_levels"] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[geom["num_levels"]]))
      
      #key_list[k] = "int"
    else:
      feat = geom[k]
      
      if (feat.dtype.char in np.typecodes["AllFloat"]):
        feat = tf.convert_to_tensor(feat.astype(np.float32), dtype=tf.float32)
        key_list[k] = ["float", geom[k].shape]
        
      else:
        if (k == "texture" or k == "image"):
          feat = tf.convert_to_tensor(feat.astype(np.uint8), dtype=tf.uint8)
          key_list[k] = ["uint8", geom[k].shape]
        else:
          feat = tf.convert_to_tensor(feat.astype(np.int32), dtype=tf.int32)
          key_list[k] = ["int", geom[k].shape]
      
      feat_serial = tf.io.serialize_tensor(feat)
      

       
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat_serial.numpy()]))
      
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString(), key_list


def load_tfr_dataset(files, ordered=True, compression_type=None):
  
 
  ignore_order = tf.data.Options()
  if not ordered:
    ignore_order.deterministic = False 
 
  #random.shuffle(files)
  if compression_type is None:
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

  else:
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE, compression_type=compression_type)
                                      
  dataset = dataset.with_options(ignore_order)
  
  return dataset


'''
///////////////////////////////////////////////////////////////////////////////
////////////////////////////// USE FOR SHAPES ///////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////
'''

#  num_elements = dataset.cardinaltiy().numpy()


def load_shape_tfrecords(path, compression_type=None):

  shape_keys = load_dict(path + "/shape_keys.json")

  files = Glob(path + "/*.tfrecords")

  #print(files, flush=True)
  #print(files, flush=True)
  #print(shape_keys, flush=True)
  
  return load_tfr_dataset(files, compression_type=compression_type), shape_keys

def get_des(key_list):
  shape_des = {}
  
  for k in key_list:
    shape_des[k] = tf.io.FixedLenFeature([], tf.string)
  
  return shape_des
  
  
def parse_data(ex, key_list, shape_des):
    
  example = tf.io.parse_single_example(ex, shape_des)
  shape = {}

  for k in key_list:
    dat = example[k]
    
    if key_list[k][0] == "string":
      feat = tf.io.parse_tensor(dat, tf.string)
      feat = tf.ensure_shape(feat, key_list[k][1])
      
    elif key_list[k][0] == "float":
      feat = tf.io.parse_tensor(dat, tf.float32)
      feat = tf.ensure_shape(feat, key_list[k][1])

    else:
      
      if (key_list[k][0] == "uint8"):
        feat = tf.io.parse_tensor(dat, tf.uint8)
      else:
        feat = tf.io.parse_tensor(dat, tf.int32)
      
      feat = tf.ensure_shape(feat, key_list[k][1])

    shape[k] = feat
    
  return shape

def parser(key_list):
  
  shape_des = get_des(key_list)
  
  return lambda ex : parse_data(ex, key_list, shape_des)

  
def identity(ex):
  return ex   


def get_num_repeat(num_el, num_steps, batch_dim=1):
  
  
  num_repeat = (num_steps * batch_dim // (num_el)) + 1 
  
  return num_repeat
  
'''
///////////////////////////////////////////////////////////////////////////////
////////////////////////////// USE FOR IMAGES ///////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////
'''


def parse_image(ex):
  
  im = ex["image"]
  
  im_rescale = tf.cast(im, tf.float32) / 255.0 
  
  return {"image": im, "im_rescale": im_rescale}

def resize_512(ex):
  im = ex["image"]
  
  im = tf.image.resize(im, [512, 512], method='nearest')
  
  return {"image":im}

def crop_512(ex):
   im = ex["image"]
   
   im = tf.image.resize_with_crop_or_pad(im, 512, 512)
   
   return {"image":im}


 
 
'''
=====================================================================
============================ Transforms =============================
=====================================================================
'''

def unit_images( example ):
  example["image"] = tf.cast( example["image"], tf.float32 ) / 255.0 
  
  return example 


def dual_images( key="image" ):
    
  def _dual_images( example, key ):
    example[key] = 2.0 * ((tf.cast(example[key], tf.float32) / 255.0) - 0.5) 
    
    return example 
  
  return lambda example: _dual_images( example, key )

def zbound(stats, std_factor = 2.5, key="image"):

  minV = stats[0] - std_factor * stats[1] 
  maxV = stats[0] + std_factor * stats[1]
  
  def _zbound(example):
    im = example[key]
    im = (im -  tf.broadcast_to(minV, im.shape) ) / (tf.broadcast_to(maxV - minV, im.shape))
    im = tf.clip_by_value(im, clip_value_min=0.0, clip_value_max=1.0)
    im = 2.0 * im - 1.0 

    example[key] = im 
    return example 

  return lambda example: _zbound(example) 
  
def resize_images(size):
  
  def _resize_images(example, size):
    example["image"] = tf.image.resize(example["image"], size) 
    return example 
    
  return lambda example: _resize_images(example, size) 


def pad_images(target_size):

  def _pad_images(example, target_size):

    image = example["image"]
    
    diff_H = (target_size[0] - image.shape[-3]) // 2 
    diff_W = (target_size[1] - image.shape[-2]) // 2 
  
    example["image"] = tf.image.pad_to_bounding_box(image, diff_H, diff_W, target_size[0], target_size[1])
    return example
    
  return lambda example: _pad_images(example, target_size) 
  
  
  
def scale_translation(scale):

  def _scale_translation(scale, example):
    g = example["g"]
    
    gS = tf.concat([g[..., :3, :3], scale * g[..., :3, -1, None]], axis=-1)
    gS = tf.concat([gS, g[..., -1, None, :]], axis=-2)
     
    example["g"] = gS
    
    return example 
    
  return lambda example: _scale_translation(scale, example)
  
  

