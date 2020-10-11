import torch
from torch.nn.functional import soft_margin_loss
from torch import optim, nn
from settings import DEVICE, ROOT, SAVED_MODEL_DIR
import numpy as np
import os.path as op
import os
from dataset import ssl_collate

def load_losses(saved_models_dir, name):
	with open(op.join(saved_models_dir, name + '_train_losses.npy'), 'rb') as f:
		train_losses = list(np.load(f))
	with open(op.join(saved_models_dir, name + '_test_losses.npy'), 'rb') as f:
		test_losses = list(np.load(f))
	return train_losses, test_losses

def save_losses(train_losses, test_losses, saved_models_dir, name):
	with open(op.join(saved_models_dir, name + '_train_losses.npy'), 'wb') as f:
		np.save(f, train_losses)
	with open(op.join(saved_models_dir, name + '_test_losses.npy'), 'wb') as f:
		np.save(f, test_losses)


def rp_loss(model, x, y):
	out = model(x)
	return soft_margin_loss(out, y)
 
def _train(model, train_loader, optimizer, epoch):
	model.train()
	
	train_losses = []
	for (anchors, sampled), y in train_loader:
		x = (anchors.to(DEVICE).to(float).contiguous(), sampled.to(DEVICE).to(float).contiguous())
		y = y.to(DEVICE).to(float).contiguous()
		loss = rp_loss(model, x, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())

	return train_losses, np.mean(train_losses)

def _eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (anchors, sampled), y in data_loader:
            x = (anchors.to(DEVICE).to(float).contiguous(), sampled.to(DEVICE).to(float).contiguous())
            y = y.to(DEVICE).to(float).contiguous()
            loss = rp_loss(model, x, y)
            total_loss += loss * x[0].shape[0] # 
        avg_loss = total_loss / data_loader.sampler.size
    return avg_loss.item()

def _train_epochs(model, train_loader, test_loader, train_args):

	epochs, lr = train_args['epochs'], train_args['lr']
	optimizer = optim.Adam(model.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 20, gamma = 0.01)
	if not os.path.exists(SAVED_MODEL_DIR):
		os.makedirs(SAVED_MODEL_DIR)
	train_losses = []
	test_losses = [_eval_loss(model, test_loader)]
	for epoch in range(1, epochs+1):
		model.train()
		losses, train_loss = _train(model, train_loader, optimizer, epoch)
		train_losses.extend([train_loss])
		test_loss = _eval_loss(model, test_loader)
		test_losses.append(test_loss)
		scheduler.step()
		print(f'Epoch {epoch},Train_loss {train_loss :.4f}, Test loss {test_loss:.4f}')
		if epoch % 2 == 0:
			torch.save(model.state_dict(), os.path.join(ROOT, 'saved_models', 'ssl_model_epoch{}.pt'.format(epoch)))
	torch.save(model.state_dict(), os.path.join(ROOT, 'saved_models', 'ssl_model.pt'))

	return train_losses, test_losses


def train_ssl(model, train_dataset, test_dataset,sampler, n_epochs=20, lr=1e-3, batch_size=256, load_last_saved_model=False, num_workers=8):
#	C = train_dataset.__getitem__((0, 0,1))[0][0].shape[0] # num channels
#	T = train_dataset.__getitem__((0, 0,1))[0][0].shape[1] # num timepoints
	
	if load_last_saved_model:
		model.load_state_dict(torch.load(os.path.join(ROOT, SAVED_MODEL_DIR, 'ssl_model.pt')))

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
        
	model.to(DEVICE)
    
   

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,sampler = sampler["train"], collate_fn=ssl_collate)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,sampler = sampler["val"], collate_fn=ssl_collate)


	new_train_losses, new_test_losses = _train_epochs(model, train_loader, test_loader, 
																				 dict(epochs=n_epochs, lr=lr))

	if load_last_saved_model:
		train_losses, test_losses = load_losses(SAVED_MODEL_DIR, 'ssl')
	else:
		train_losses = []
		test_losses = []
	
	train_losses.extend(new_train_losses)
	test_losses.extend(new_test_losses)

	save_losses(train_losses, test_losses, SAVED_MODEL_DIR, 'ssl')

	return train_losses, test_losses, model
