

checkpoint_directory = '/home/atlas/PycharmProjects/pythorchvisiontransformer/bestmodels/'
checkpoint_filepath = checkpoint_directory + 'pnemothorax_best_model.pt'
DIRECTORY_PATH = '/home/atlas/datasets/rostami/pneumothorax/'
RGB_DIRECTORY_PATH = '/home/atlas/datasets/rostami/pneumothorax_rgb/'
AUGMENTED_PATH= '/home/atlas/datasets/rostami/augmentedpneumothorax/'
CLAHE_PATH = '/home/atlas/datasets/rostami/clahe_pnemothorax/'
RGB_CLAHE_PATH = '/home/atlas/datasets/rostami/clahe_pnemothorax_rgb/'

BALANCED_PATH = '/home/atlas/datasets/rostami/balancedpneumothorax/'
RGB_BALANCED= '/home/atlas/datasets/rostami/balancedpneumothorax_rgb/'
PREPROCESSED_BALANCED = '/home/atlas/datasets/rostami/preprocessedbalancedpneumothorax/'
RGB_PREPROCESSED_BALANCED = '/home/atlas/datasets/rostami/preprocessedbalancedpneumothorax_rgb/'
AUGMENTED_BALANCED = '/home/atlas/datasets/rostami/preprocessedbalancedneumothorax/'
MODELS_FOLDER = '/home/atlas/PycharmProjects/visiontransformer/savedmodels/'
RESULTS_XCELL = '/home/atlas/PycharmProjects/dqn/results.xlsx'
MINIMAL_RGB = '/home/atlas/datasets/rostami/minimal_rgb/'



TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'
VALIDATION_FOLDER = 'validation'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELES = 3
NORMAL_CLASS = 0
PNEMOTHORAX_CLASS = 1
NUM_CLASSES = 2
BATCH_SIZE = 3
RANDOM_SEED = 17
PATCH_SIZE = 7
EPOCHS =10
input_state_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELES)
