from yacs.config import CfgNode as CN


_C = CN()
_C.PIPELINE_TYPE = None

_C.VOICE2POSE = CN()
_C.VOICE2POSE.STRICT_LOADING = True
_C.VOICE2POSE.GENERATOR = CN()
_C.VOICE2POSE.GENERATOR.NAME = None
_C.VOICE2POSE.GENERATOR.LEAKY_RELU = True
_C.VOICE2POSE.GENERATOR.NORM = 'IN'
_C.VOICE2POSE.GENERATOR.LAMBDA_REG = 1.0
_C.VOICE2POSE.GENERATOR.LAMBDA_CLIP_KL = 0.1
_C.VOICE2POSE.GENERATOR.CLIP_CODE = CN()
_C.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION = None
_C.VOICE2POSE.GENERATOR.CLIP_CODE.LR_SCALING = 1.0
_C.VOICE2POSE.GENERATOR.CLIP_CODE.TRAIN = True
_C.VOICE2POSE.GENERATOR.CLIP_CODE.FRAME_VARIANT = False
_C.VOICE2POSE.GENERATOR.CLIP_CODE.SAMPLE_FROM_NORMAL = False
_C.VOICE2POSE.GENERATOR.CLIP_CODE.TEST_WITH_GT_CODE = False
_C.VOICE2POSE.GENERATOR.CLIP_CODE.EXTERNAL_CODE = False
_C.VOICE2POSE.GENERATOR.CLIP_CODE.EXTERNAL_CODE_PTH = None

_C.VOICE2POSE.POSE_ENCODER = CN()
_C.VOICE2POSE.POSE_ENCODER.NAME = 'PoseSeqEncoder'
_C.VOICE2POSE.POSE_ENCODER.AE_CHECKPOINT = None

_C.VOICE2POSE.POSE_DISCRIMINATOR = CN()
_C.VOICE2POSE.POSE_DISCRIMINATOR.NAME = None
_C.VOICE2POSE.POSE_DISCRIMINATOR.LEAKY_RELU = False
_C.VOICE2POSE.POSE_DISCRIMINATOR.LAMBDA_GAN = 1.0
_C.VOICE2POSE.POSE_DISCRIMINATOR.MOTION = True
_C.VOICE2POSE.POSE_DISCRIMINATOR.WHITE_LIST = None

_C.POSE2POSE = CN()
_C.POSE2POSE.AUTOENCODER = CN()
_C.POSE2POSE.AUTOENCODER.NAME = None
_C.POSE2POSE.AUTOENCODER.LEAKY_RELU = True
_C.POSE2POSE.AUTOENCODER.NORM = 'BN'
_C.POSE2POSE.AUTOENCODER.CODE_DIM = 32
_C.POSE2POSE.LAMBDA_REG = 1.0
_C.POSE2POSE.LAMBDA_KL = 0.1

_C.DATASET = CN()
_C.DATASET.NAME = 'GestureDataset'
_C.DATASET.ROOT_DIR = 'datasets/speakers'
_C.DATASET.SUBSET = None
_C.DATASET.NUM_LANDMARKS = 533
_C.DATASET.HIERARCHICAL_POSE = True
_C.DATASET.SPEAKER = None
_C.DATASET.NUM_FRAMES = 64
_C.DATASET.AUDIO_LENGTH = 68267
_C.DATASET.MAX_DEMO_LENGTH = 24  # seconds
_C.DATASET.AUDIO_SR = 16000  # audio sampling rate
_C.DATASET.FPS = 15
_C.DATASET.CACHING = False

_C.TRAIN = CN()
_C.TRAIN.NUM_EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SAVE_VIDEO = True
_C.TRAIN.SAVE_NPZ = True
_C.TRAIN.LR = 1e-4
_C.TRAIN.WD = 0
_C.TRAIN.LR_SCHEDULER = True
_C.TRAIN.PRETRAIN_FROM = None
_C.TRAIN.VALIDATE = True
_C.TRAIN.NUM_RESULT_SAMPLE = 2
_C.TRAIN.CHECKPOINT_INTERVAL = 1  # Interval of epochs for checkpoint saving

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32
_C.TEST.NUM_RESULT_SAMPLE = 8
_C.TEST.SAVE_VIDEO = True
_C.TEST.SAVE_NPZ = True
_C.TEST.MULTIPLE = 1

_C.DEMO = CN()
_C.DEMO.MULTIPLE = 1
_C.DEMO.NUM_SAMPLES = 1
_C.DEMO.CODE_INDEX = None
_C.DEMO.CODE_INDEX_B = None
_C.DEMO.CODE_PATH = None

_C.SYS = CN()
_C.SYS.OUTPUT_DIR = 'output/'
_C.SYS.CANVAS_SIZE = (720, 1280)
_C.SYS.VISUALIZATION_SCALING = 0.85
_C.SYS.VIDEO_FORMAT = ['mp4', 'img']  # ['tensorboard', 'mp4', 'img']
_C.SYS.ASYNC_VIDEO_SAVING = False
_C.SYS.LOG_INTERVAL = 100  # Interval of steps for logging
_C.SYS.NUM_WORKERS = 8
_C.SYS.DISTRIBUTED = False
_C.SYS.WORLD_SIZE = 1
_C.SYS.MASTER_ADDR = 'localhost'
_C.SYS.MASTER_PORT = 21379

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  return _C.clone()
