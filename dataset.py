from pylab import *
import torch 
from settings import DEVICE, STIMULUS_IDS, RAW_DIR, MASTOID_REC
import numpy as np
import os
import json
import mne
import contextlib
import io
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

class WeightedSampler(torch.utils.data.sampler.Sampler):
    r"""Sample des windows randomly
    Arguments:
    ---------
        dataset (Dataset): dataset to sample from
        size (int): The total number of sequences to sample
    """

    def __init__(self,dataset, batch_size,size,  weights):
    
        
        self.batch_size = batch_size
        self.size = size
        self.dataset = dataset
        self.serie_len = 0
        self.n_subjects = len(self.dataset.subjects)
        self.weights = torch.DoubleTensor(weights)
        with open(os.path.join(RAW_DIR, f"recordings_info_1.json"),'r') as json_file:self.lenghts = json.load(json_file)
        
    def __iter__(self):
        num_batches = self.size// self.batch_size
        n_subject_samples = self.batch_size //self.n_subjects
        while num_batches > 0:
            #
            if num_batches % 100 == 0 :
                print("batches restants :",num_batches)
            #iterate on each subject in the dataset
            for subject in self.dataset.subjects:
                sampled = 0
                self.serie_len = self.lenghts[subject][1]
                #sample `n_subject_samples` per subject
                while sampled < n_subject_samples:
                    # each sample is a target and an anchor window. Positive or/and negative windows are sampled in the Dataset class. 
                    target  = 2*torch.multinomial(
                self.weights, 1, replacement=True) -1
                    t = choice(arange(0, self.serie_len-self.dataset.temp_len, 1))
                    sampled += 1
                    yield (t,target,subject)
            
            num_batches -=1

    def __len__(self):
        return len(self.dataset)   


class Abstract_Dataset(torch.utils.data.Dataset):
    '''
    Classe dataset  pour les differents sampling
    '''
    def __init__(self, subjects, temp_len , n_features):
        self.subjects = subjects
        self.time_series = []
        self.temp_len = temp_len
        self.n_features = n_features
    def get_windows(self,index):
        '''
        a method to load  a sequence 
        '''
        raise NotImplementedError
    def get_pos(self, t_anchor):
        '''
        a method to get positive samples
        '''
        raise NotImplementedError
    def load_ts(self, index):
        '''
        a method to get positive samples
        '''
        raise NotImplementedError
    def get_neg(self, t_anchor):
        '''
       a method to get negative samples
        '''
        raise NotImplementedError
    def get_targets(self, index):
        '''
        a method to get labels
        '''
        raise NotImplementedError
    def __getitem__(self, index):
        windows = self.get_windows(index)
        target = self.get_targets(index)
        return windows, target
    def __len__(self): return self.time_series.shape[1]

class RP_Dataset(Abstract_Dataset):
    
    def __init__(self, time_series, sampling_params, temp_len , n_features ):
      super().__init__(time_series, temp_len = temp_len, n_features = n_features)
      self.pos , self.neg = sampling_params
    def get_windows(self,index):
        '''
        a method to get sampled windows
        '''
        (t, target,subject) = index
#        print(subject)
        #load ts
        del self.time_series# to save memory
        self.time_series =self.load_ts(subject)
        # slice 
        anchor_wind = self.time_series[:self.n_features,t:t+self.temp_len]
        # sample a positive or negative window
        t_ = self.get_pos(t) if target>0 else self.get_neg(t)
        sampled_wind = self.time_series[:self.n_features,t_:t_+self.temp_len] # could be negative or positive
        return (anchor_wind, sampled_wind)
    
    def load_ts(self, subject):
        path_processed= os.path.join(RAW_DIR, f"{subject}-processed.npy")
        return np.load(path_processed,mmap_mode = "c")
    def get_targets(self, index):
        return index[1]
    def get_pos(self, t_anchor):

      start = max(0,t_anchor-self.pos ) 
      end = min(self.__len__()-self.temp_len-1,t_anchor+self.pos-1 ) # to get a sequence of lenght self.temp_lenght
      t_ = choice(arange(start,end, 1)) 
      return t_
    def get_neg(self, t_anchor):
      
      left_idx = arange(0, max(0, t_anchor - self.neg), 1)
      right_idx =arange(min(self.__len__()-self.temp_len-1, t_anchor + self.neg-1),self.__len__()-self.temp_len-1 ,1)
      t_ = choice(hstack([left_idx, right_idx]))
      return t_

def ssl_collate(batch):

    anchors = torch.stack([torch.from_numpy(item[0][0]) for item in batch])
    try:
        sampled = torch.stack([torch.from_numpy(item[0][1]) for item in batch])
    except:
        print("error")
    targets = torch.stack([item[1] for item in batch])
    
    return (anchors, sampled), targets

class DecoderSampler(torch.utils.data.sampler.Sampler):
    r"""Sample des windows randomly
    Arguments:
    ---------
        dataset (Dataset): dataset to sample from
        size (int): The total number of sequences to sample
    """

    def __init__(self,dataset, batch_size,size,  weights):
    
        
        self.batch_size = batch_size
        self.size = size
        self.dataset = dataset
        self.weights = torch.DoubleTensor(weights)
        self.n_subjects = len(self.dataset.subjects)

    def __iter__(self):
        num_batches = self.size// self.batch_size
        n_subject_samples = self.batch_size //self.n_subjects
        while num_batches > 0:
            #sample a subject 
            batch_subjects = choice(self.dataset.subjects,replace = False, size =5)
            for subject in  batch_subjects:# TODO sample subjects ?
                sampled = 0
                #for each subject sample `n_subject_samples`  couples (target class, trial)
                while sampled < n_subject_samples:
                    indice  = torch.multinomial(
                self.weights, 1, replacement=True)
                    trial = choice(self.dataset.trials)
                    sampled += 1
                    yield ( indice,trial, subject)
                
            num_batches -=1

    def __len__(self):
        return len(self.size)   

class SequentialSampler(torch.utils.data.sampler.Sampler):
    r"""sampler for validation iterate sequentialy on subjects 
    Arguments:
    ---------
        dataset (Dataset): dataset to sample from
        size (int): The total number of sequences to sample
    """

    def __init__(self,dataset, batch_size,size,  weights):
    
        
        self.batch_size = batch_size
        self.size = size
        self.dataset = dataset
        self.weights = torch.DoubleTensor(weights)
        self.n_subjects = len(self.dataset.subjects)
        
    def __iter__(self):
        num_batches = self.size// self.batch_size
        n_subject_samples = self.batch_size //self.n_subjects
        while num_batches > 0:
            sampled = 0
            while sampled < self.batch_size:   
                for subject in self.dataset.subjects:# TODO sample subjects ?
                    for indice in range(len(self.weights)):
                        for trial in self.dataset.trials:                
                                sampled += 1
                                yield ( indice,trial, subject)
                
            num_batches -=1

    def __len__(self):
        return len(self.size)
class  Decoder_Dataset(torch.utils.data.Dataset):
    '''
    Classe dataset  pour les differents sampling
    '''
    def __init__(self, subjects, trials, T , step, condition = 1):

        self.subjects = subjects
        self.temp_len = T
        self.step = step
        self.trials = trials
        self.condition = condition
    def load_epoch(self, index):
        #print(index)
        ( indice,trial, subject) = index
        stim_id = STIMULUS_IDS[indice]
        
        epochs_path = os.path.join(RAW_DIR, f"{subject}-epochs_{stim_id}_{self.condition}-epo.fif")
        with nostdout():
            if subject in MASTOID_REC:
                epochs = mne.read_epochs(epochs_path, verbose = False)[trial].get_data()
            else :
                epochs = mne.read_epochs(epochs_path, verbose = False)[trial].drop_channels(['EXG5','EXG6']).get_data()
        
        return epochs
    def __getitem__(self, index):
        classe = STIMULUS_IDS[index[0]]
        sampled = self.load_epoch(index)
        x_sample = torch.tensor(sampled).unfold(-1, self.temp_len, self.step ).permute(2,0,1,3) # .to(DEVICE)
        return x_sample, index[0]
    def __len__(self): len(self.subjects)*len(self.trials )*12
        
def collate(batch):
   
    samples= [item[0] for item in batch]
    
    targets = [item[1] for item in batch]
    
    return samples, targets

