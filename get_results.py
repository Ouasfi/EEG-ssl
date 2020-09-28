import numpy as np 
from train  import load_losses
from settings import SAVED_MODEL_DIR

if __name__ == '__main__':
    train_losses, test_losses = load_losses(SAVED_MODEL_DIR, 'ssl')
    print(test_losses)

