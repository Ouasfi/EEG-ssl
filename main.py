from models import Relative_Positioning, StagerNet
from dataset import RP_Dataset, WeightedSampler
from settings  import *
from preprocessing import *
from train import train_ssl
import argparse

parser = argparse.ArgumentParser('Parameter tuning for classification on a defined dataset')
parser.add_argument('--path', type=str, default='none')
parser.add_argument('--subject', type=str, default='01')
parser.add_argument('--C', help= "number of features", type=int, default=69)
parser.add_argument('--T', help = "temporal lenght of the sampled windows",type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--pos', help= "positive samples ",  type=int, default=600)
parser.add_argument('--neg', help= "negative samples ",  type=int, default=800)
parser.add_argument('--w',help= "proportion of positive samples in the dataset", type=float, default=0.5)
parser.add_argument('--lr',help = " learning rate", type=float, default= 1e-3)
parser.add_argument('--n_train',help = "number of sampled windows in the training set", type=float, default= 3000)
parser.add_argument('--n_test',help = "number of sampled windows in the training set", type=float, default= 300)
parser.add_argument('--resume',help = "load_last_saved_model", type=bool, default= False)





args = parser.parse_args()
#subejct
subject = args.subject
# models params
C = args.C
T =args.T
#training paramms
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
resume = args.resume
#datasets params
n_train = args.n_train
n_test = args.n_test
#sampling params
pos= args.pos
neg = args.pos

s_weights = [args.w, 1-args.w]

# EEG data of subject 01
if __name__ == '__main__':
    print( "Training parameters:")
    print("======================")
    print(f"- subject: {subject}\n- C : {C}\n- T : {T}\n- epochs : {epochs}\n- batch size : {batch_size}\n- lr : {lr}\n- n_train : {n_train}\n- n_test : {n_test}\n- pos : {pos}\n- neg : {neg}")
    
    X = process(subject)
    #split data
    split = int(X.shape[1]*0.6)
    X_train = X[:, :split]
    X_test = X[:, split:]
    #define ssl model
    ssl_model = Relative_Positioning(StagerNet,C , T )
    ssl_model.to(float)
    # datasets
    train_dataset =  RP_Dataset(X_train, sampling_params = (pos, neg), temp_len = T ,
                                n_features = C )
    test_dataset =  RP_Dataset(X_test, sampling_params = (pos, neg), temp_len = T ,
                                n_features = C )

    train_sampler = WeightedSampler(train_dataset, batch_size = batch_size ,size = n_train,  
                              weights = s_weights)
    test_sampler = WeightedSampler(test_dataset, batch_size = batch_size ,size = n_test,  
                              weights = s_weights)
    samplers = {"train" : train_sampler, "val": test_sampler}


    #train ssl 
    train_losses, test_losses, model = train_ssl(ssl_model, train_dataset, test_dataset,
                                                 samplers,n_epochs=epochs, lr=lr,batch_size= batch_size, 
                                                 load_last_saved_model=False, num_workers= 0)

