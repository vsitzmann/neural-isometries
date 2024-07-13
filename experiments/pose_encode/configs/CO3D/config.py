# Logging Info
WORK_DIR = "" # Leave empty
PROJECT_NAME = "NISO_POSE_ENCODE"


# LOSS_WEIGHTS 
ALPHA_EQUIV = 5.0e-1
BETA_MULT = 2.5e-2 

# Latent parameters
KERNEL_OP_DIM = 128 # Rank of operator, equivalent to number of learned eigenfunctions
LATENT_DIM  = 128

MAX_STEP    = 12 # Max frameskip during training


BATCH_SIZE      = 2
TRUE_BATCH_SIZE = 8 


DATASET_NAME = "CO3D"
INPUT_SIZE = (144, 144) 


DATA_COMPRESSION_TYPE = None
DATA_TRAIN_SIZE = 10000 
DATA_TEST_SIZE =  2000


## Encoder + Decoder params 
CONV_ENC_CHANNELS    = (64, 128, 256)
CONV_ENC_BLOCK_DEPTH = (4, 4, 4)

CONV_DEC_CHANNELS    = (256, 128, 64)
CONV_DEC_BLOCK_DEPTH = (4, 4, 4) 

KERNEL_SIZE = 3

# Hyperparameters

ADAM_B1 = 0.9 #0.5
ADAM_B2 = 0.999 #0.9 

NUM_TRAIN_STEPS = 1000000
STOP_AFTER      = 200000
INIT_LR         = 1.0e-8 
LR              = 5.0e-4 
END_LR          = 5.0e-5
WARMUP_STEPS    = 2000 

EVAL_EVERY     = 10000
LOG_LOSS_EVERY = 100
VIZ_EVERY      = 2500
VIZ_SIZE       = INPUT_SIZE
NUM_EVAL_STEPS = 10

CHECKPOINT_EVERY = 10000

TRIAL = 0  # Dummy for repeated runs.

