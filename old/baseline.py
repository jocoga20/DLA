import math

import torch
from tqdm import tqdm
import wandb
from model_pipelines import *
from Pipeline import Pipeline
from utils import *
from metrics import *

class CBR(Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2)
		self.bn = torch.nn.BatchNorm2d(num_features=out_channels)

	def forward(self, x):
		x = self.conv(x)
		x1 = x
		x = self.bn(x) + x1
		x = torch.nn.functional.relu(x)
		return x



def main():
	num_classes = 43
	epochs = 500
	trainloader, testloader = default_loaders(train_batch_size=512, test_batch_size=128)
	dim = 16
	model = torch.nn.Sequential(CBR(3, dim), CBR(dim, 2*dim), CBR(2*dim, 4*dim), torch.nn.Flatten(), torch.nn.LazyLinear(out_features=num_classes)).cuda()
	optimizer = torch.optim.SGD(params=[p for p in model.parameters() if p.requires_grad], lr=1e-3, momentum=0)

	wandb.login(key='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')

	wbrun = wandb.init(entity='jonathangallicoli', project='DLA', name='baseline', mode='online', dir='wandb_tmp')
	scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

	p = Pipeline(model, criterion=torch.nn.CrossEntropyLoss(), optimizer=optimizer, scheduler=scheduler)
	
	chance_level_mean_loss = math.log(num_classes)
	chance_level_accuracy = 1 / num_classes
	print(f'Avg loss: {chance_level_mean_loss}')
	print(f'Avg acc:  {chance_level_accuracy}')
	best_acc = chance_level_accuracy
	best_checkpoint_dict = None
	
	try:
		for e in tqdm(range(epochs)):
			train_loss, train_acc = p.train_epoch(trainloader)
			scheduler.step()
			with torch.no_grad():
				test_loss, test_acc = p.eval_epoch(testloader)
			if test_acc > best_acc:
				best_checkpoint_dict = p.get_checkpoint()
				

			wbrun.log({
				'loss_train': train_loss,
				'acc_train': train_acc,
				'loss_test': test_loss,
				'acc_test': test_acc,
				'loss_diff': loss_diff(train_loss, test_loss),
				'acc_diff': accuracy_diff(train_acc, test_acc),
				'lr': scheduler.get_last_lr()[0]
			})
	finally:
		del trainloader
		del testloader
		save_path = f'checkpoints/cbr.ep{e}.acc{round(test_acc*100)}.pt'
		print(f'Saving {save_path}')
		p.checkpoint = best_checkpoint_dict
		p.save_checkpoint(save_path)
		wbrun.finish()

if __name__ == '__main__':
	main()
