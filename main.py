from models import Relative_Positioning, StagerNet
from dataset import RP_Dataset, WeightedSampler
from settings  import *
from preprocessing import *
from train import train_ssl


C = 69
T = 200

# EEG data of subject 01
X = process("01")
#split data
split = int(X.shape[1]*0.6)
X_train = X[:, :split]
X_test = X[:, split:]
#define ssl model
ssl_model = Relative_Positioning(StagerNet,C , T )
ssl_model.to(float)
# datasets
train_dataset =  RP_Dataset(X_train, sampling_params = (600, 1000), temp_len = T ,
                            n_features = C )
test_dataset =  RP_Dataset(X_test, sampling_params = (600, 1000), temp_len = T ,
                            n_features = C )
                        
train_sampler = WeightedSampler(train_dataset, batch_size = 30 ,size = 1000,  
                          weights = [0.5, 0.5])
test_sampler = WeightedSampler(test_dataset, batch_size = 30 ,size = 300,  
                          weights = [0.5, 0.5])
samplers = {"train" : train_sampler, "val": test_sampler}


#train ssl 
train_losses, test_losses, model = train_ssl(ssl_model, train_dataset, test_dataset,
                                             samplers,n_epochs=15, lr=1e-3,batch_size=10, 
                                             load_last_saved_model=False, num_workers= 0)

