"""
Default configurations for action recognition domain adaptation
"""


from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.ROOT = ""
_C.DATASET.NAME = "ASPIRE"
_C.DATASET.SRC_TARGETS =  "/shared/tale2/Shared/data/CMRI/ASPIRE/cardiac4ch_labels_VPnC_CV"
_C.DATASET.IMAGE_MODALITY = "CMRI"
_C.DATASET.LANDMARKS = []
_C.DATASET.TRAINSET_SIZE = -1  # -1 for full trainset size or int <= len(training_set)



_C.SAMPLER = CN()
_C.SAMPLER.DEBUG = False
_C.SAMPLER.PATCH_SIZE = [64, 64]
_C.SAMPLER.RESOLUTION_TO_SAMPLE_FROM = [512,512]  # ['full', 'input_size']

_C.SOLVER = CN()
_C.SOLVER.SEED = 42
_C.SOLVER.BASE_LR = 0.01  # Initial learning rate
_C.SOLVER.DECAY_POLICY = "poly"  # ["poly", None]
_C.SOLVER.MAX_EPOCHS = 1000
_C.SOLVER.DATA_LOADER_BATCH_SIZE = 12
_C.SOLVER.NUM_RES_SUPERVISIONS = 5
_C.SOLVER.DEEP_SUPERVISION = True
_C.SOLVER.LOSS_FUNCTION = "mse"  


## model trainer
_C.TRAINER = CN()
_C.TRAINER.PERFORM_VALIDATION = True
_C.TRAINER.SAVE_LATEST_ONLY = True
_C.TRAINER.CACHE_DATA = True
_C.TRAINER.FOLD = 0
_C.TRAINER.INFERENCE_ONLY = True


# ---------------------------------------------------------------------------- #
# U-Net Model configs# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "U-Net"  
_C.MODEL.GAUSS_SIGMA = 4
_C.MODEL.HM_LAMBDA_SCALE = 100.0
_C.MODEL.CHECKPOINT = None

_C.MODEL.UNET = CN()
_C.MODEL.UNET.MIN_FEATURE_RESOLUTION = 4
_C.MODEL.UNET.MAX_FEATURES = 512
_C.MODEL.UNET.INIT_FEATURES = 32



_C.INFERENCE = CN()
_C.INFERENCE.EVALUATION_MODE = "scale_heatmap_first"  # ["scale_heatmap_first", "scale_pred_coords", "use_input_size"]

_C.INFERENCE.DEBUG = False
# --------------------------------
# -------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = True
_C.OUTPUT.OUTPUT_DIR = "/output/"
_C.OUTPUT.OUTPUT_LOG_NAME = "/output_log.log"
_C.OUTPUT.USE_COMETML_LOGGING = False

_C.OUTPUT.COMET_API_KEY = None
_C.OUTPUT.COMET_WORKSPACE = "default"
_C.OUTPUT.COMET_PROJECT_NAME = "landnnunet"

_C.OUTPUT.COMET_TAGS = ["default"]
_C.OUTPUT.RESULTS_CSV_APPEND = None



def get_cfg_defaults():
    return _C.clone()
