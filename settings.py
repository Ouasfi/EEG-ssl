
import os
import torch

DIR = os.path.split(os.getcwd())[0]
ROOT = os.getcwd()
SUBJECTS = ['P01','P04','P06','P07','P09',
            'P11','P12', 'P13','P14']# TODO add 'P05' when ica is computed

MASTOID_REC =['Pilot3','P01','P02','P03','P04','P05','P06','P07','P08']
CONDITIONS = ['cued', 'non-cued', 'free']
KEYSTROKE_BASE_ID = 2000
STIMULUS_IDS = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
v_1_subjects =  ['Pilot3','P01','P04','P05','P06','P07'] 

RAW_DIR = os.path.join(DIR,"openmiir/raw_data/")
ICA_DIR = os.path.join(DIR,"openmiir/eeg/preprocessing/ica/")
METADATA_DIR = os.path.join(DIR,"openmiir/")
DATA_PATH = METADATA_DIR

SAVED_MODEL_DIR = "saved_models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_VERSION = 2
VIEW = -1e-6
