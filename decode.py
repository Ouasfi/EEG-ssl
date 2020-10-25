import numpy as np # linear algebra
import os
from settings  import *
from preprocessing import *
from models import Relative_Positioning, StagerNet, Decoder
from dataset import Decoder_Dataset, DecoderSampler, collate
import process
from train import load_losses, save_losses
from torch import nn, optim
import argparse 
from get_results import *
def _train_dec(model, train_loader, optimizer, epoch):
    
    model.train()
    train_losses = []
    batch_size = train_loader.batch_size
    for batch_x, batch_y in train_loader:
        #print(batch_y)
        #print('in')
        loss = torch.tensor([0.0], requires_grad=True)
        for x,y in zip(batch_x, batch_y):
            x = x.squeeze(dim= 0 ).to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            #print(out.shape)
            loss = loss+ model.loss_fn(out.unsqueeze(dim =0), y)
        loss =loss/batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return [mean(train_losses)]
def _eval_loss_dec(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.squeeze(dim= 0 ).to(DEVICE)
            y = y.to(DEVICE)
            loss = model.loss_fn(model(x).unsqueeze(dim = 0), y)
            total_loss += loss #* x[0].shape[0] #
        avg_loss = total_loss / data_loader.sampler.size# / len(data_loader.dataset)
    return avg_loss.item()

def _train_epochs_dec(model, train_loader, test_loader, train_args):
    
    epochs, lr = train_args['epochs'], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 10, gamma = 0.1)
    if not os.path.exists(SAVED_MODEL_DIR):
        os.makedirs(SAVED_MODEL_DIR)

    train_losses = []
    test_losses = [_eval_loss_dec(model, test_loader)]
    for epoch in range(1, epochs+1):
        model.train()
        train_losses.extend(_train_dec(model, train_loader, optimizer, epoch))
        test_loss = _eval_loss_dec(model, test_loader)
        test_losses.append(test_loss)
        y_true, y_pred = get_test_results(model, test_loader)
        acc_score = accuracy_score(y_true, y_pred)
        scheduler.step()
        print(f'Epoch {epoch}, Train loss {train_losses[-1]:.4f},Test loss {test_loss:.4f}, \tAccuracy: {100*acc_score:.2f}%')

        # save model every 10 epochs
        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(ROOT, 'saved_models', 'decoder_epoch{}.pt'.format(epoch)))
    torch.save(model.state_dict(), os.path.join(ROOT, 'saved_models', 'decoder.pt'))
    return train_losses, test_losses



def train_decoder(model, train_dataset, val_dataset,samplers, n_epochs=20, lr=1e-3, batch_size=256, load_last_saved_model=False, num_workers=8):
	
	if load_last_saved_model:
		model.load_state_dict(torch.load(os.path.join(ROOT, SAVED_MODEL_DIR, 'decoder.pt')))
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.to(DEVICE)

	train_loader = train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size, num_workers=0,
                                          sampler = samplers["train"], collate_fn = collate)
	val_loader = val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0,
                                          sampler = samplers["val"])
	new_train_losses, new_test_losses = _train_epochs_dec(model, train_loader, val_loader, 
																				 dict(epochs=n_epochs, lr=lr))
	if load_last_saved_model:
		train_losses, test_losses = load_losses(SAVED_MODEL_DIR, 'decoder')
	else:
		train_losses = []
		test_losses = []
	train_losses.extend(new_train_losses)
	test_losses.extend(new_test_losses)
	save_losses(train_losses, test_losses, SAVED_MODEL_DIR, 'decoder')
	return train_losses, test_losses, model
def freeze(model):

    for param in model.parameters():
        param.requires_grad = False
    return model
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parameter tuning for classification on a defined dataset')
    parser.add_argument('--path', type=str, default='none')
    parser.add_argument('--subject', type=str, default='01')
    parser.add_argument('--C', help= "number of features", type=int, default=64)
    parser.add_argument('--T', help = "temporal lenght of the sampled windows",type=int, default=1000)
    parser.add_argument('--M', help = "Embedding dimension",type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--w',help= "proportion of positive samples in the dataset", type=float, default=0.5)
    parser.add_argument('--lr',help = " learning rate", type=float, default= 1e-3)
    parser.add_argument('--n_train',help = "number of sampled windows in the training set", type=float, default= 3000)
    parser.add_argument('--n_val',help = "number of sampled windows in the validation set", type=float, default= 300)
    parser.add_argument('--resume',help = "load_last_saved_model", type=bool, default= False)


    args = parser.parse_args()
    #subejct
    subject = args.subject
    # models params
    C = args.C
    T =args.T
    M = args.M
    #training paramms
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    resume = args.resume
    #datasets params
    n_train = args.n_train
    n_val= args.n_val
    #sampling params

    subjects = SUBJECTS[:-2]
    #split data
    test_subjects = [SUBJECTS[-2]]
    # load embedding model
    ssl_model = Relative_Positioning(StagerNet,C , T, embedding_dim = M )
    ssl_model.load_state_dict(torch.load(os.path.join(ROOT, 'saved_models', 'ssl_model.pt')))

    model = freeze(ssl_model.feature_extractor) #Freeze extractor weights for transfer learning
    
    #data_dict = process.get_features('01', condition = 1)
    aggregator = torch.mean
    decoder = Decoder(model, aggregator, C, T, embedding_dim=M, hidden_dim = 20)
    decoder.to(float).to(DEVICE)
    train_dataset = Decoder_Dataset(subjects,[0,1,2], T, step = 512)
    val_dataset = Decoder_Dataset(subjects,[3,4], T, step = 512)
    train_sampler = DecoderSampler(train_dataset, batch_size = 12, weights = [1/12]*12, size = 65)
    val_sampler = DecoderSampler(val_dataset, batch_size = 12, weights = [1/12]*12, size = 65)
    samplers = {"train" : train_sampler, "val": val_sampler}
    train_losses, test_losses, model = train_decoder(decoder, train_dataset,val_dataset,samplers, n_epochs=epochs, lr=lr)
