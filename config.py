import torch
import torchvision.transforms as transforms

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
BATCH_SIZE = 128
NUM_WORKER = 0
NUM_EPOCHS = 1000
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_REC = 0.999
LAMBDA_ADV = 0.001
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_G = "pretained_model/context_encoder_generatior.pth.tar"
CHECKPOINT_D = "pretained_model/context_encoder_discriminator.pth.tar"

transform = transforms.Compose(
    [
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)