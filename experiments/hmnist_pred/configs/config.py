# Logging Info
WORK_DIR = "" # Leave empty
PROJECT_NAME = "NISO_HMNIST_PRED"


HMNIST_DIR = ''

PRE_TRAIN_DIR = ''

# Vector Neuron Network params
MLP_FEATURES =128 
MLP_LAYERS = 2 


# Latent parameters
KERNEL_OP_DIM = 32
LATENT_DIM  = 32


BATCH_SIZE      = 16
TRUE_BATCH_SIZE = 16 
MAX_BATCH_SIZE  = 64

INPUT_SIZE = (40, 40) #RE10K
DATASET_NAME = "mnist"

DATA_COMPRESSION_TYPE = 'None' #None
DATA_TRAIN_SIZE = 60000 // 2 
DATA_TEST_SIZE = 10000 // 2 
# Training parameters



## Encoder + Decoder params 
CONV_ENC_CHANNELS    = (64, 128)
CONV_ENC_BLOCK_DEPTH = (3, 3)

CONV_DEC_CHANNELS    = (128, 64)
CONV_DEC_BLOCK_DEPTH = (3, 3) 

KERNEL_SIZE = 3

## Hyperparameters

ADAM_B1 = 0.9 #0.9# 0.5
ADAM_B2 = 0.999 #0.999 # 0.9 

NUM_TRAIN_STEPS = 1000000
STOP_AFTER      = 20000
INIT_LR         = 1.0e-8 
LR              = 5.0e-4 
END_LR          = 5.0e-5
WARMUP_STEPS    = 2000 

EVAL_EVERY     = 20000
LOG_LOSS_EVERY = 100
VIZ_EVERY      = 2500
VIZ_SIZE       = INPUT_SIZE
NUM_EVAL_STEPS = 500 

CHECKPOINT_EVERY = 10000


TRIAL = 0  # Dummy for repeated runs.

