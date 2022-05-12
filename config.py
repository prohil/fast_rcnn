# ROOT_PATH = 'input/FLIR/train/'
# TEST_PATH = 'input/FLIR/test/'
# ROOT_PATH = 'checkpoints/400 photos/train/'
# TEST_PATH = 'checkpoints/400 photos/test/'
# ROOT_PATH = 'checkpoints/4000 photos/train/'
# TEST_PATH = 'checkpoints/4000 photos/test/'
ROOT_PATH = 'checkpoints/FLIR filtered/train/'
TEST_PATH = 'checkpoints/FLIR filtered/test/'
TRAIN_ANNOTATIONS = f"input/FLIR/train/train_annotations > 700.csv"

PREDICTION_THRES = 0.8
EPOCHS = 1
MIN_SIZE = 300
BATCH_SIZE = 2
DEBUG = False  # to visualize the images before training