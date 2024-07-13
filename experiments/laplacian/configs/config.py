# Logging Info
WORK_DIR = "" # Leave empty
PROJECT_NAME = "NISO_LAPLACIAN"


# Transformation parameters
BETA_MULT = 5.0e-1

GRID_DIM = (16, 16)
LATENT_DIM = GRID_DIM[0] * GRID_DIM[1] + 10 

# Latent parameters
KERNEL_OP_DIM = GRID_DIM[0] * GRID_DIM[1] 


BATCH_SIZE      = 16
TRUE_BATCH_SIZE = 16 
MAX_BATCH_SIZE  = 64

INPUT_SIZE = (256, 256)


## Hyperparameters
ADAM_B1 = 0.9 # 0.5
ADAM_B2 = 0.999 # 0.9 

NUM_TRAIN_STEPS = 1000000
STOP_AFTER      = None
INIT_LR         = 1.0e-8 
LR              = 5.0e-4 
END_LR          = 5.0e-5
WARMUP_STEPS    = 2000 

EVAL_EVERY     = 5000
LOG_LOSS_EVERY = 100
VIZ_EVERY      = 500
VIZ_SIZE       = (1024, 1024)
NUM_EVAL_STEPS = 500 

CHECKPOINT_EVERY = 10000

TRIAL = 0  # Dummy for repeated runs.
