
import os
import torch

SUBJECTS = ['P01','P02','P03','P04','P05','P06','P07','P08','P09','P10',
            'P11','P12']

recording_with_mastoid_channels =['Pilot3','P01','P02','P03','P04','P05','P06','P07','P08']
CONDITIONS = ['cued', 'non-cued', 'free']
KEYSTROKE_BASE_ID = 2000
STIMULUS_IDS = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
v_1_subjects =  ['Pilot3','P01','P04','P05','P06','P07'] 

RAW_DIR = "/kaggle/input/rawsbeforehoes/"
ICA_DIR = '/kaggle/input/openmiir/eeg/preprocessing/ica/'
METADATA_DIR = "/kaggle/input/openmiir"
DATA_PATH = METADATA_DIR
ROOT = os.getcwd()
SAVED_MODEL_DIR = "saved_models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_VERSION = 2