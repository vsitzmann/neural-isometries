import os
import numpy as np
import scipy as sp
from math import pi as PI
import json
import plyfile 
from PIL import Image 



Open = open 
MakeDirs = os.mkdir 


## Reading meshes adapted from PyTorch Geometric

####################################
########## OBJ Files ###############
####################################

## Adapted from https://github.com/pyg-team/pytorch_geometric/blob/66b17806b1f4a2008e8be766064d9ef9a883ff03/torch_geometric/io/obj.py

def yield_file_for_obj(file_name):
  file = Open(file_name, 'rt')
  buffer = file.read()
  file.close()
  for row in buffer.split('\n'):
    if row.startswith('v '):
      yield ['v', [float(x) for x in row.split(' ')[1:]]]
    elif row.startswith('f '):
      
      triangles = row.split(' ')[1:]
    
      fv = [] 
      ft = [] 
      
      for t in triangles:
        t_s = t.split('/') 
        fv.append(int(t_s[0]) - 1)
        if (len(t_s) > 0):
          ft.append(int(t_s[1]) - 1)
          
      if not ft: 
        yield ['f', fv]
      else:
        yield ['f', (fv, ft)]

    elif row.startswith('vt '):
      yield ['vt', [float(x) for x in row.split(' ')[1:]]]
    else:
      yield ['', '']


def load_obj(file_name):
  vertices = []
  faces = []
  coords = []
  for k, v in yield_file_for_obj(file_name):
    if k == 'v':
      vertices.append(v)
    elif k == 'f':
      faces.append(v)
    elif k == 'vt':
      coords.append(v) 
      
  if not faces or not vertices:
    return None

  if not coords:
    return np.asarray(vertices, dtype=np.double), np.asarray(faces, dtype=np.int_)
  else:   
    V = np.asarray(vertices, dtype=np.double) 
    
    F = [] 
    CU = [] 
    CV = []
    
    c_ind = []
    for l in range(len(faces)):
      F.append(faces[l][0])
      fu = []
      fv = []
      for j in range(3):
        fu.append(coords[faces[l][1][j]][0])
        fv.append(coords[faces[l][1][j]][1])
      
   
      CU.append(fu)
      CV.append(fv)
        
    CU = np.asarray(CU, dtype=np.double)
    CV = np.asarray(CV, dtype=np.double)    
  
    return V, np.asarray(F, dtype=np.int_), np.concatenate(( CU[..., None], CV[..., None]), axis=-1)
  

####################################
########## OFF Files ###############
####################################

## Adapted from https://github.com/pyg-team/pytorch_geometric/blob/66b17806b1f4a2008e8be766064d9ef9a883ff03/torch_geometric/io/off.py

def load_off(file_name):

  with Open(file_name, 'rt') as f:
    file = f.read().split('\n')[:-1]

  if file[0] == 'OFF':
    file = file[1:]
  else:
    file[0] = file[0][3:]

  num_vertices, num_faces = [int(item) for item in file[0].split()[:2]]

  vertices = np.squeeze(
      np.asarray([[float(x)
                   for x in line.split(None)[0:None]]
                  for line in file[1:1 + num_vertices]],
                 dtype=np.double))

  facets = file[1 + num_vertices:1 + num_vertices + num_faces]

  facets = [[int(x) for x in line.strip().split()] for line in facets]

  faces = np.asarray([line[1:] for line in facets if line[0] == 3],
                     dtype=np.int_)

  # Convert non-triangular faces to triangles
  recs = [line[1:] for line in facets if line[0] == 4]

  if recs:
    recs = np.asarray(recs, dtype=np.int_)

    faces = np.concatenate((faces, recs[:, [0, 1, 2]], recs[:, [0, 2, 3]]))

  return vertices, faces


####################################
########## PLY Files ###############
####################################


def load_ply(file_name):

  with Open(file_name, 'rb') as f:
    mesh = plyfile.PlyData.read(f)

  vertices_x = np.asarray(mesh['vertex']['x'], dtype=np.double)
  vertices_y = np.asarray(mesh['vertex']['y'], dtype=np.double)
  vertices_z = np.asarray(mesh['vertex']['z'], dtype=np.double)

  vertices = np.concatenate(
      (vertices_x[:, None], vertices_y[:, None], vertices_z[:, None]), axis=-1)

  faces = None

  if 'face' in mesh:
    faces = mesh['face']['vertex_indices']
    faces = [np.asarray(fc, dtype=np.int_) for fc in faces]
    faces = np.stack(faces, axis=-1)

  return vertices, np.transpose(faces, (1, 0))

def write_ply(vertices, faces, file_name, face_uv = None):
  # face_uv: F x 3 x 2 
  v_list = []
  for l in range(vertices.shape[0]):
    v_list.append((vertices[l, 0], vertices[l, 1], vertices[l, 2]))
  
  f_list = faces.tolist() 
  
  vertex_data = np.array(v_list, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]) 
  V = plyfile.PlyElement.describe(vertex_data, 'vertex')
  
  if face_uv is None:
    face_data = np.empty(faces.shape[0], dtype=[('vertex_indices', 'i4', (3, ))])
  else:
    face_data = np.empty(faces.shape[0], dtype=[('vertex_indices', 'i4', (3, )), ('texcoord', 'f4', (6, ))])
    
    c_list = np.concatenate((face_uv[:, 0, :], face_uv[:, 1, :], face_uv[:, 2, :]), axis=-1).tolist() 
    face_data['texcoord'] = np.array(c_list, dtype='f4')
    
  face_data['vertex_indices'] = np.array(f_list, dtype='i4')
  
  F = plyfile.PlyElement.describe(face_data, 'face')
  if (file_name[-4:] != '.ply'):
    plyfile.PlyData([V, F], text=True).write(str(file_name + '.ply'))
  else:
    plyfile.PlyData([V, F], text=True).write(str(file_name))
    
  
def load_masked_ply(file_name):

  with Open(file_name, 'rb') as f:
    mesh = plyfile.PlyData.read(f)

  vertices_x = np.asarray(mesh['vertex']['x'], dtype=np.double)
  vertices_y = np.asarray(mesh['vertex']['y'], dtype=np.double)
  vertices_z = np.asarray(mesh['vertex']['z'], dtype=np.double)

  vertices = np.concatenate(
      (vertices_x[:, None], vertices_y[:, None], vertices_z[:, None]), axis=-1)

  faces = None

  if 'face' in mesh:
    faces = mesh['face']['vertex_indices']
    faces = [np.asarray(fc, dtype=np.int_) for fc in faces]
    faces = np.stack(faces, axis=-1)

  mask = np.asarray(mesh['face']['quality'], dtype=np.int_)
  
  return vertices, np.transpose(faces, (1, 0)), mask 

####################################
########### Read mesh ##############
####################################


def load_mesh(file_name):

  ext = os.path.splitext(file_name)[1]

  if ext == '.ply':
    return load_ply(file_name)
  elif ext == '.obj':
    return load_obj(file_name)
  elif ext == '.off':
    return load_off(file_name)
  else:
    print(
        'ERROR: File passed to load_mesh is not a .ply, .obj, or an .off file!',
        flush=True)
    return None, None

#####################################
########### Load NPZ archive ########
#####################################

def load_npz(file_name, pickle=True):
  
  direc = {} 
  
  with Open(file_name, 'rb') as f, np.load(f, allow_pickle=pickle) as npz_archive:
    for k in npz_archive:
      direc[k] = npz_archive[k]
      
  return direc 

def load_npy(file_name):
  with  Open(file_name, 'rb') as f:
    out = np.load(f)
    
  return out

def load_txt(file_name):
  
  with Open(file_name, 'rb') as f:
    out = np.loadtxt(f)
  
  return out 

def load_png(file_name):
  
  with Open(file_name, 'rb') as f:
    out = np.array(Image.open(f))
  
  return out 

def save_png(x, file_name):
  
  #MakeDirs(os.path.dirname(file_name))

  with Open(file_name, 'wb') as f:
    Image.fromarray(x).save(f)
      
#########################################
######## Save + load dictionary files ###
#########################################

def save_dict(file_name, in_dict):
  
  json_dict = json.dumps(in_dict)

  with Open(file_name, "w") as f:
    f.write(json_dict)
    f.close()
  

def load_dict(file_name):
  
  with Open(file_name) as json_file:
    data = json.load(json_file)
  
  return data 




