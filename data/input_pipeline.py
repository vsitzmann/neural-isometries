import sys
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy as sp
import numpy as np 

from os.path import dirname, abspath
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.append( ROOT_PATH )

from data import data_loader as dl 
from icecream import ic 


def get_mnist():
  train = tfds.load( 'mnist', split='train', shuffle_files=True )
  test  = tfds.load( 'mnist', split='test' )

  return train, test 


def get_hmnist( hmnist_file ):
  hmnist = sp.io.loadmat( hmnist_file)
  
  images = (255.0 * hmnist["img_data"]).astype( np.float32 )
  labels = hmnist["labels"].astype( np.int32 )
  
  shuffle_ind = np.random.permutation( images.shape[0] )
  
  num_test = images.shape[0] 

  images = images[..., None]
  
  images = tf.convert_to_tensor(images)
  labels = tf.convert_to_tensor(labels)
  
  test = tf.data.Dataset.from_tensor_slices( {"image": images, "label": labels} )
  
  return test, num_test


def get_cshrec11(cshrec11_dir):
  NUM_CLASSES = len(os.listdir(cshrec11_dir + "train")) - 1 
  
  train_data, train_keys = dl.load_shape_tfrecords( os.path.join(cshrec11_dir, "train") )
  test_data, test_keys   = dl.load_shape_tfrecords( os.path.join(cshrec11_dir, "test") )
      
  train_data = train_data.map( dl.parser(train_keys), num_parallel_calls=tf.data.AUTOTUNE )
  test_data  = test_data.map( dl.parser(test_keys), num_parallel_calls=tf.data.AUTOTUNE )

  # Load stats 
  stats = np.load(os.path.join(cshrec11_dir, "stats.npz"))

  mean = stats["MEAN"]
  mean2 = stats["MEAN2"]

  std = np.sqrt(mean2 - mean ** 2) 

  return train_data, test_data, NUM_CLASSES, (tf.convert_to_tensor(mean, dtype=tf.float32), tf.convert_to_tensor(std, dtype=tf.float32))


def get_seq_dataset( dataset_dir, compression_type=None ):
  train_data, train_keys = dl.load_shape_tfrecords( dataset_dir + "train", compression_type=compression_type )
  test_data, test_keys   = dl.load_shape_tfrecords( dataset_dir + "test", compression_type=compression_type )
  
  train_data = train_data.map( dl.parser(train_keys), num_parallel_calls=tf.data.AUTOTUNE )
  test_data  = test_data.map( dl.parser(test_keys), num_parallel_calls=tf.data.AUTOTUNE )
  
  return train_data, test_data 
  
