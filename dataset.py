from pylab import *
import torch 

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
        self.serie_len = len(self.dataset)
        
        self.weights = torch.DoubleTensor(weights)
        
    def __iter__(self):
        num_batches = self.size// self.batch_size
        while num_batches > 0:
            #print()
            sampled = 0
            while sampled < self.batch_size:
                target  = 2*torch.multinomial(
            self.weights, 1, replacement=True) -1
                t = choice(arange(0, self.serie_len-self.dataset.temp_len, 1))
                t_ = self.dataset.get_pos(t) if target>0 else self.dataset.get_neg(t)
                sampled += 1
                yield (t, t_, target)
            
            num_batches -=1

    def __len__(self):
        return len(self.train_list)   


class Abstract_Dataset(torch.utils.data.Dataset):
    '''
    Classe dataset  pour les differents sampling
    '''
    def __init__(self, time_series, temp_len , n_features):

        self.time_series = time_series
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
        (t, t_ , _) = index
        anchor_wind = self.time_series[:,t:t+self.temp_len]
        neg_wind = self.time_series[:,t_:t_+self.temp_len] # could be negative or positive
        return (anchor_wind, neg_wind)
    
    def get_targets(self, index):
        return index[-1]
    def get_pos(self, t_anchor):

      start = max(0,t_anchor-self.pos ) 
      end = min(self.__len__()-self.temp_len,t_anchor+self.pos ) # to get a sequence of lenght self.temp_lenght
      t_ = choice(arange(start,end, 1)) 
      return t_
    def get_neg(self, t_anchor):
      
      left_idx = arange(0, max(0, t_anchor - self.neg), 1)
      right_idx =arange(min(self.__len__()-self.temp_len, t_anchor + self.neg),self.__len__()-self.temp_len ,1)
      t_ = choice(hstack([left_idx, right_idx]))
      return t_

def collate(batch):

    anchors = torch.stack([torch.from_numpy(item[0][0]) for item in batch])
    try:
        sampled = torch.stack([torch.from_numpy(item[0][1]) for item in batch])
    except:
        print(batch)
    targets = torch.stack([item[1] for item in batch])
    
    return (anchors, sampled), targets

