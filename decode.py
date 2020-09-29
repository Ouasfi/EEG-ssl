import numpy as np # linear algebra
import os
from settings  import *
from preprocessing import *
from models import Relative_Positioning, StagerNet, Decoder
from dataset import Decoder_Dataset, Decoder_Dataset
import process
from torch import nn

def _train_dec(model, train_loader, optimizer, epoch):
    
	model.train()
	train_losses = []
    
	for x, y in train_loader:
		
		y = y.to(DEVICE).to(float).contiguous()
		loss = model.loss_fn(model(x), y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())
	return train_losses
def _train_epochs_dec(model, train_loader, test_loader, train_args):
	epochs, lr = train_args['epochs'], train_args['lr']
	optimizer = optim.Adam(model.parameters(), lr=lr)
	if not os.path.exists(SAVED_MODEL_DIR):
		os.makedirs(SAVED_MODEL_DIR)
	
	train_losses = []
	test_losses = [_eval_loss_dec(model, test_loader)]
	for epoch in range(1, epochs+1):
		model.train()
		train_losses.extend(_train_dec(model, train_loader, optimizer, epoch))
		test_loss = _eval_loss_dec(model, test_loader)
		test_losses.append(test_loss)
		print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
		
		# save model every 10 epochs
		if epoch % 2 == 0:
			torch.save(model.state_dict(), os.path.join(ROOT, 'saved_models', 'decoder_epoch{}.pt'.format(epoch)))
	torch.save(model.state_dict(), os.path.join(ROOT, 'saved_models', 'decoder.pt'))
	return train_losses, test_losses

def _eval_loss_dec(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            loss = model.loss_fn(model(x), y)
            total_loss += loss * x[0].shape[0] # 
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()

def train_decoder(model, train_dataset, test_dataset,sampler, n_epochs=20, lr=1e-3, batch_size=256, load_last_saved_model=False, num_workers=8):
	
	
	if load_last_saved_model:
		model.load_state_dict(torch.load(os.path.join(ROOT, SAVED_MODEL_DIR, 'decoder.pt')))

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
        
	model.to(DEVICE)
    
   

	train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0,sampler = sampler)
	test_loader = torch.utils.data.DataLoader(test, num_workers=0,sampler = sampler)
	new_train_losses, new_test_losses = _train_epochs_dec(model, train_loader, test_loader, 
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parameter tuning for classification on a defined dataset')
    parser.add_argument('--path', type=str, default='none')
    parser.add_argument('--subject', type=str, default='01')
    parser.add_argument('--C', help= "number of features", type=int, default=69)
    parser.add_argument('--T', help = "temporal lenght of the sampled windows",type=int, default=1000)
    parser.add_argument('--M', help = "Embedding dimension",type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=15)
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
    M = args.M
    #training paramms
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    resume = args.resume
    #datasets params
    n_train = args.n_train
    n_test = args.n_test
    #sampling params
    

    ssl_model = Relative_Positioning(StagerNet,C , T, embedding_dim = M )
    ssl_model.load_state_dict(torch.load(os.path.join(ROOT, 'saved_models', 'ssl_model.pt')))
    model = ssl_model.feature_extractor

    aggregator = torch.mean
    decoder = Decoder(model, aggregator, C, T, embedding_dim=M, hidden_dim = 20)
    decoder.to(float).to(DEVICE)
    dataset = Decoder_Dataset(data_dict, T, step = 512)
    sampler = DecoderSampler(dataset, batch_size = 12, weights = [1/12]*12, size = 65)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0,sampler = sampler)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0,sampler = sampler)
    losses = []
    for x, y in loader:
        x = x.squeeze(dim= 0 )
        y = y.to(DEVICE)
        out = decoder(x)
        loss = decoder.loss_fn(out.unsqueeze(dim = 0), y)
        print(loss.item())