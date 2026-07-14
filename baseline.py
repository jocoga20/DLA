import math

import torch
from tqdm import tqdm
import wandb
from model_pipelines import *
from Training import Training
from utils import *
from metrics import *

def main():
	num_classes = 43
	epochs = 500
	trainloader, testloader = default_loaders(train_batch_size=2048, test_batch_size=128)
	model = torchvision.models.resnet18()
	
	optimizer = torch.optim.SGD(params=[p for p in model.parameters() if p.requires_grad], lr=1e-5, momentum=0)

	wandb.login(key='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')

	myrun = Training(model=model, optimizer=optimizer)
	wbrun = wandb.init(entity='jonathangallicoli', project='DLA', name='baseline', mode='online', dir='wandb_tmp')
	scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
	best_acc = 0.39
	chance_level_mean_loss = math.log(num_classes)
	chance_level_accuracy = 1 / num_classes
	print(f'Avg loss: {chance_level_mean_loss}')
	print(f'Avg acc:  {chance_level_accuracy}')

	try:
		for e in tqdm(range(epochs)):
			train_loss, train_acc = myrun.train(loader=trainloader)
			scheduler.step()
			with torch.no_grad():
				test_loss, test_acc = myrun.eval(loader=testloader)
			if test_acc > best_acc:
				save_path = f'models/resnet18.ep{e}.acc{round(test_acc*100)}.pt'
				print(f'Saving {save_path}')
				torch.save(model.state_dict(), save_path)

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
		
		wbrun.finish()

if __name__ == '__main__':
	main()
