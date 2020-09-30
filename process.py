from preprocessing import *
from settings import METADATA_DIR, STIMULUS_IDS, VIEW
import numpy as np
import json
def decode_event(event_id):
    stimulus_id = event_id // 10
    condition = event_id % 10
    return stimulus_id, condition

def get_stim_events(events, stimulus_ids='all', conditions='all'):
    filtered = []
    for event in events:
        event_id = event[2]
        if event_id >= 1000:
            continue

        stimulus_id, condition = decode_event(event_id)

        if (stimulus_ids == 'all' or stimulus_id in stimulus_ids) and \
            (conditions == 'all' or condition in conditions):
            
            filtered.append(event)

    return np.array(filtered)

import mne
def get_trial_epochs(raw, trial_events, stim_id, condition,
                     subject=None, stimuli_version=None, meta=None,
                     include_cue=False, picks=None, debug=False):

    assert subject is None or stimuli_version is None or meta is None

    if meta is None:
        if stimuli_version is None:
            if subject is None:
                raise RuntimeError('Either meta, stimuli_version or subject has to be specified.')
            else:
                stimuli_version = get_stimuli_version(subject)
        meta = load_stimuli_metadata(data_root = METADATA_DIR, version=stimuli_version)

    events = get_stim_events(trial_events, [stim_id], [condition])
    
    #print(events)
    start = VIEW
    if condition in [1,2]: # cued
        if include_cue:
            stop = meta[stim_id]['length_with_cue']
        else:
            # NOTE: start > 0 does not work; need to shift event time
            offset = int(np.floor(meta[stim_id]['length_of_cue'] * raw.info['sfreq']))
            events[:,0] += offset
            stop = meta[stim_id]['length_without_cue']
    else:
        stop = meta[stim_id]['length_without_cue']
    print(start, stop)
    if picks is None:
        # default: all EEG channels including bad/interpolated
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])

    epochs = mne.Epochs(raw, events, None,
                              tmin=start, tmax=stop, preload=True,
                              proj=False, picks=picks, verbose=False)

    
    return epochs

def get_features(subject,condition):
    
    raw = get_raw(subject)
    ica = get_ica(subject)
    signal = ica.apply(raw)
    events = get_events(raw)
    data_dict = {}
    for stim_id in STIMULUS_IDS:
        epochs = get_trial_epochs(signal, events, stim_id = stim_id, condition= condition,
                     subject=subject, stimuli_version=None, meta=None,
                     include_cue=True, picks=None, debug=False)
        data_dict[stim_id]= epochs   
    return data_dict
def visualise_epochs(subject, signal, stim_id, condition = 1):

    events = get_events(signal)
    epochs = get_trial_epochs(signal, events, stim_id = stim_id, condition= condition,
                     subject=subject, stimuli_version=None, meta=None,
                     include_cue=True, picks=None, debug=False)
    epochs.plot(n_epochs = 10)

if __name__ == '__main__':
    condition = 1
    shapes = {}
    for subject in SUBJECTS:
        
        subject = subject[1:]
        raw = get_raw(subject)
        ica = get_ica(subject)
        reconstructed = ica.apply(raw)
        picks = mne.pick_types(reconstructed.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])
        eeg_signal = reconstructed.get_data(picks)
        path_processed= os.path.join(RAW_DIR, f"P{subject}-processed.npy")
        shapes[subject] = eeg_signal.shape
        np.save(path_processed,eeg_signal )
        events = get_events(reconstructed)
        for stim_id in STIMULUS_IDS:

            epochs = get_trial_epochs(reconstructed, events, stim_id = stim_id, condition= condition,
                     subject=subject, stimuli_version=None, meta=None,
                     include_cue=True, picks=picks, debug=False)
            epochs_name = os.path.join(RAW_DIR, f"P{subject}-epochs_{stim_id}_{condition}-epo.fif")
                     
            epochs.save(epochs_name, overwrite=True)
            print("saving file", epochs_name)

    with open(os.path.join(RAW_DIR, f"recordings_info_{condition}.json", 'w') as json_file: json.dump(shapes, json_file)    
