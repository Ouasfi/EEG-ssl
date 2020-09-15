

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
import mne
from mne import read_proj
from mne.io import read_raw_fif
from pylab import *
from mne.preprocessing import read_ica, ICA
import xlrd
from  settings import *







def get_raw(subject):
    """
    a function to get the raw data corresponding to a subject ID
    parameters:
    -----------
    - subject : str 
    correspond to the ID of the subject
    returns: mne RAW object 
    -------
    """
    path_raw = os.path.join(RAW_DIR, f"P{subject}-raw.fif")
    return  mne.io.read_raw_fif(path_raw,preload=True)
def get_ica(subject):
    """
    a function to get the ica corresponding to a subject ID
    parameters:
    -----------
    - subject : str
    correspond to the ID of the subject
    returns: mne ICA object 
    -------
    """

    path_ica = os.path.join(ICA_DIR, f'{subject}-100p_64c-ica.fif')
    return read_ica(path_ica)
def reconstuct_signal(raw, ica):
    """
    a function to get the reconstruct eeg signal from raw using the computed ICs 
    parameters:
    -----------
    - raw : mne RAW object
    - ica : mne ICA object

    
    returns: numpy array
    -------
    """
    signal = ica.apply(raw)
    return signal.get_data()
    
    
    
def process(subject):
    """
    a function to get preprocessed eeg signal for a single subject

    """
    # Bandpass filtring 
    raw = get_raw(subject)
    ica = get_ica(subject)
    
    return reconstuct_signal(raw, ica)


def get_events(raw):
    
    return  mne.find_events(raw, stim_channel='STI 014')


def get_stimuli_version(subject):
    version = 1 if  subject in ['Pilot3','P01','P04','P05','P06','P07'] else 2
    return version
    

def get_audio_filepath(stim_id, data_root=None, version=None):

    if version is None:
        version = DEFAULT_VERSION

    if data_root is None:
        data_root = os.path.join(DATA_PATH, 'OpenMIIR')

    meta = load_stimuli_metadata(data_root=data_root, version=version)

    return os.path.join(data_root, 'audio', 'full.v{}'.format(version),
                        meta[stim_id]['audio_file'])

def load_stimuli_metadata(data_root=None, version=None, verbose=None):

    if version is None:
        version = DEFAULT_VERSION

    if data_root is None:
        data_root = os.path.join(DATA_PATH, 'OpenMIIR')

    xlsx_filepath = os.path.join(data_root, 'meta', 'Stimuli_Meta.v{}.xlsx'.format(version))
    book = xlrd.open_workbook(xlsx_filepath, encoding_override="cp1252")
    sheet = book.sheet_by_index(0)

    if verbose:
        log.info('Loading stimulus metadata from {}'.format(xlsx_filepath))

    meta = dict()
    for i in range(1, 13):
        stimulus_id = int(sheet.cell(i,0).value)
        meta[stimulus_id] = {
            'id' : stimulus_id,
            'label' : sheet.cell(i,1).value.encode('ascii'),
            'audio_file' : sheet.cell(i,2).value.encode('ascii'),
            'cue_file' : sheet.cell(i,2).value.replace('.wav', '_cue.wav'),
            'length_with_cue' : sheet.cell(i,3).value,
            'length_of_cue' : sheet.cell(i,4).value,
            'length_without_cue' : sheet.cell(i,5).value,
            'length_of_cue_only' : sheet.cell(i,6).value,
            'cue_bpm' : int(sheet.cell(i,7).value),
            'beats_per_bar' : int(sheet.cell(i,8).value),
            'num_bars' : int(sheet.cell(i,14).value),
            'cue_bars' : int(sheet.cell(i,15).value),
            'bpm' : int(sheet.cell(i,16).value),
            'approx_bar_length' : sheet.cell(i,11).value,
        }

        if version == 2:
            meta[stimulus_id]['bpm'] = meta[stimulus_id]['cue_bpm'] # use cue bpm

    return meta


def target_to_path(target_id, subject, mapping_dict):
    
    version = 1 if  subject in ['Pilot3','P01','P04','P05','P06','P07'] else 2
    mapping = load_stimuli_metadata(data_root=METADATA_DIR, version=version, verbose=None) 
        

    return os.path.join(METADATA_DIR, 'audio', 'full.v{}'.format(version),
                        mapping[target_id]['audio_file'].decode("utf-8") )

def data_from_raw(raw):
    return raw[:-1][0]

def get_perception_data(eeg_data, events ):
    
    """
    a function to get perception features and target ids from eeg 
    data recording of a single subject using an events array
    
    """
    
    start = 0
    features = []
    target_ids = []
    
    for event in events:
        end , event_id = event[0], event[2]
        
        if event_id < 1000: #Event Ids < 1000 are trial labels
            stimulus_id = event_id // 10
            condition = event_id % 10
            
            if condition == 1: # 1: preception trials
                features.append(eeg_data[:,start:end])
                target_ids.append(stimulus_id)
                
        start = end 
    return {"features": features, "targets": target_ids}