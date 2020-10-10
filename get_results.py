import numpy as np 
from train  import load_losses
from settings import SAVED_MODEL_DIR
from dataset import *
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
from torch import nn
import numpy as np
from torch.utils import data

def get_ssl_results(model, test_loader):
	y_true = []
	y_pred = []
	model.eval()
	softmax = nn.Softmax()
	with torch.no_grad():
		for (anchors, sampled), y in test_loader:
                    x = (anchors.to(DEVICE).to(float).contiguous(), sampled.to(DEVICE).to(float).contiguous())
                    y = y.to(DEVICE).to(float).contiguous()
                    out = model(x)
                    _, predicted = torch.max(softmax(out.data), 1)
                    y_true.extend(list(y.cpu().numpy()))
                    y_pred.extend(list(predicted.cpu().numpy()))
    return y_true, y_pred
def get_ssl_scores(model, test_loader):
    y_true, y_pred = get_ssl_results(model, test_loader)
    acc_score = accuracy_score(y_true, y_pred)
    balanced_acc_score = balanced_accuracy_score(y_true, y_pred)
    print(f'\tAccuracy: {100*acc_score:.2f}%')
    print(f'\tBalanced accuracy: {100*balanced_acc_score:.2f}%')
    return acc_score, balanced_acc_score

def get_test_results(model, test_loader):
    y_true = []
    y_pred = []
    model.eval()
    softmax = nn.Softmax(dim = 1)
    with torch.no_grad():
        
        for x, y in test_loader:
            
            x = x.to(DEVICE).to(float).contiguous().squeeze(dim= 0)
            y = y.to(DEVICE).to(float).contiguous()
            out = model(x)
            _, predicted = torch.max(softmax(out.unsqueeze(dim =0)), 1)
            y_true.extend(list(y.cpu().numpy()))
            y_pred.extend(list(predicted.cpu().numpy()))
    return y_true, y_pred

def decoder_scores(model, subjects, trials):
    
    test_dataset = Decoder_Dataset(subjects,trials, T, step = 512)
    test_sampler = SequentialSampler(test_dataset, batch_size = 168, weights = [1/12]*12, size = 168)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0,
                                          sampler = test_sampler)
    y_true, y_pred = get_test_results(model, test_loader)
    acc_score = accuracy_score(y_true, y_pred)
    balanced_acc_score = balanced_accuracy_score(y_true, y_pred)
    print(f'Performance of the network on the test trials:')
    print(f'\tAccuracy: {100*acc_score:.2f}%')
    print(f'\tBalanced accuracy: {100*balanced_acc_score:.2f}%')
    return acc_score, balanced_acc_score

if __name__ == '__main__':
    train_losses, test_losses = load_losses(SAVED_MODEL_DIR, 'ssl')
    print(train_losses)
    print(test_losses)

