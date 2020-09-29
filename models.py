import torch
from torch.nn import *
from pylab import *
from torch import optim
from torch.utils import data
from torch import nn
from torch.nn.functional import soft_margin_loss
from settings import DEVICE

class StagerNet(Module):
    """
     StagerNet implementation.    
    """
    def __init__(self, num_classes, num_channels, temp_lenght ):
        super().__init__()      
        # create conversion layer
        self.relu = ReLU()
        self.spatial_conv = Conv2d(1, num_channels, (num_channels,1), stride= (1,1))
        self.temp_conv1 =  Conv2d(1, 16, (1,51), stride= (1,1), padding= (0,(51-1)//2))#(51-1)//2 to insure same padding
        self.batch_norm1 = BatchNorm2d(16)
        self.temp_conv2 =  Conv2d(16, 16, (1,51), stride= (1,1), padding= (0,(51-1)//2))#(51-1)//2 to insure same padding
        self.batch_norm2 = BatchNorm2d(16)
        self.maxPool = MaxPool2d((1, 13), stride=(1, 13))
        self.flatten = Flatten()
        self.dropout = Dropout(p = 0.5)
        self.linear_class = Linear(num_channels*(temp_lenght//(13*13))*16,num_classes )        
    def forward(self, inputs):
      x = self.spatial_conv(inputs)
      x = x.permute(0,2,1,3)
      x = self.relu(self.temp_conv1(x))
      # a relu activation is used before batch_norm, is it the case in the original implementation ?
      x = self.batch_norm1(x) 
      x = self.relu(x)      
      x = self.maxPool(x)
      x = self.relu(self.temp_conv2(x)) 
      x = self.batch_norm2 (x)
      x = self.relu(x)
      x = self.maxPool(x)
      x = self.dropout(self.flatten(x))
      x = self.linear_class(x)
      return x


class ShallowNet(Module):
    """
     ShallowNet implementation.    
    """
    def __init__(self, num_classes, num_channels , temp_lenght):
        super().__init__()      
        # create conversion layer
        self.eps = 1e-45
        self.relu = ReLU()
        
        self.temp_conv1 =  Conv2d(1, 40, (1,25), stride= (1,1))
        self.batch_norm2 = BatchNorm2d(40)
        self.spatial_conv = Conv2d(40, 40, (num_channels,1), stride= (1,1))
        self.meanPool = AvgPool2d((1, 75), stride=(1, 15))
        self.flatten = Flatten()
        self.dropout = Dropout(p = 0.5)
        self.num_features = (((temp_lenght-25+1)-75)//15+1)*40
        self.linear_class = Linear( self.num_features,num_classes )  

    def forward(self, inputs):
      x = self.temp_conv1(inputs)
      x = self.batch_norm2(x)
      x = self.spatial_conv(x)
      x = torch.pow(x, 2) # squaring non-linearity

      x = self.meanPool(x)
      x = self.flatten(x)
      x = torch.log(x+self.eps) #log  non-linearity 
      
      x = self.dropout(self.flatten(x))
      x = self.linear_class(x)
      return x
    """
     ShallowNet implementation.    
    """
    def __init__(self, num_classes, num_channels , temp_lenght):
        super().__init__()      
        # create conversion layer
        self.eps = 1e-45
        self.relu = ReLU()
        
        self.temp_conv1 =  Conv2d(1, 40, (1,25), stride= (1,1))
        self.batch_norm2 = BatchNorm2d(40)
        self.spatial_conv = Conv2d(40, 40, (num_channels,1), stride= (1,1))
        self.meanPool = AvgPool2d((1, 75), stride=(1, 15))
        self.flatten = Flatten()
        self.dropout = Dropout(p = 0.5)
        self.num_features = (((temp_lenght-25+1)-75)//15+1)*40
        self.linear_class = Linear( self.num_features,num_classes )  

    def forward(self, inputs):
      x = self.temp_conv1(inputs)
      x = self.batch_norm2(x)
      x = self.spatial_conv(x)
      x = torch.pow(x, 2) # squaring non-linearity

      x = self.meanPool(x)
      x = self.flatten(x)
      x = torch.log(x+self.eps) #log  non-linearity 
      
      x = self.dropout(self.flatten(x))
      x = self.linear_class(x)
      return x

class Relative_Positioning(nn.Module):
  def __init__(self, EEG_FeatureExtractor, C, T, embedding_dim=100):
    super().__init__()
    self.feature_extractor = EEG_FeatureExtractor(num_classes =embedding_dim , num_channels = C , temp_lenght = T).to(float).to(DEVICE)
    #self.feature_extractor.float()
    self.linear = nn.Linear(embedding_dim, 1)
    self.loss_fn = nn.SoftMarginLoss()

  def forward(self, x):
    first_samples = x[0].unsqueeze(dim=1)
    second_samples = x[1].unsqueeze(dim=1)

    h_first = self.feature_extractor(first_samples)
    h_second = self.feature_extractor(second_samples)

    h_combined = torch.abs(h_first - h_second)

    out = self.linear(h_combined)
    return out

class Decoder(nn.Module):
    
    def __init__(self, EEG_FeatureExtractor, aggregator,  C, T, embedding_dim=100, hidden_dim= 20):
        super().__init__()
        self.feature_extractor = EEG_FeatureExtractor
        #self.feature_extractor.float()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 12)
        self.aggr = aggregator
        self.loss_fn = nn.CrossEntropyLoss()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        out = self.aggr(x, axis = 0)
        return out   
