import torch.optim as optim
import torchvision.transforms as T

# For list of supported models use timm.list_models
MODEL_NAME = "resnet18"
NUM_ClASSES = 10
IN_CHANNELS = 3

USE_TORCHVISION = False  # If you need to use timm models set to False.

# USE_TORCHVISION = True # Should use Torchvision Models or timm models
PRETRAINED = "imagenet"  # If True -> Fine Tuning else Scratch Training
EPOCHS = 5
TRAIN_BATCH_SIZE = 512  # Training Batch Size
VALID_BATCH_SIZE = 512  # Validation Batch Size
NUM_WORKERS = 4  # Workers for training and validation

EARLY_STOPPING = True  # If you need early stoppoing for validation loss
SAVE_PATH = "{}.pt".format(MODEL_NAME)

# IMG_WIDTH = 224  # Width of the image
# IMG_HEIGHT = 224  # Height of the image
MOMENTUM = 0.8  # Use only for SGD
LEARNING_RATE = 1e-3  # Learning Rate
SEED = 42

LOG_INTERVAL = 300  # Interval to print between epoch

# Train and validation Transforms which you would like
train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

# Classes to be detected.
FASHION_MNIST_CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot",)

# Classes to be detected.
CIFAR10_CLASSES = ("airplane", "automobile", "bird",
                   "cat", "deer", "dog", "frog", "horse", "ship", "truck",)
