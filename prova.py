from Pipeline import Pipeline
import torch, torchvision
import torchvision.transforms.v2 as t
from torch.utils.data import DataLoader
from torch.nn import Module

from lab1 import set_seed

def get_cifar(is_train_split, transform):
	return torchvision.datasets.CIFAR10('./mydatasets', train=is_train_split, transform=transform, download=True)

def cifar_train(transform): return get_cifar(True, transform)
def cifar_test(transform): return get_cifar(False, transform)

def sgd_from_model(model: Module):
	return torch.optim.SGD(params=[p for p in model.parameters() if p.requires_grad], lr=1e-1)

def scheduler_from_optim(optim):
	return torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20)

def main():
	g = set_seed()
	train_transform = t.Compose([
		t.RandomHorizontalFlip(),
		t.ToImage(),
		t.ToDtype(torch.float32, scale=True),
		t.Normalize(mean=[0.5]*3, std=[0.5]*3)
	])
	test_transform = t.Compose([
		t.ToImage(),
		t.ToDtype(torch.float32, scale=True),
		t.Normalize(mean=[0.5]*3, std=[0.5]*3)
	])

	model = torchvision.models.resnet18(weights=None, num_classes=10)
	model.requires_grad_(False)
	model.conv1.requires_grad_()
	model.bn1.requires_grad_()
	model = model.cuda()
	optim = sgd_from_model(model)
	bs = 512
	epochs = 20

	trainloader = DataLoader(
		dataset=cifar_train(train_transform),
		batch_size=bs,
		shuffle=True,
		pin_memory=True,
		persistent_workers=True,
		num_workers=10,
		generator=g
	)

	testloader = DataLoader(
		dataset=cifar_test(test_transform),
		batch_size=bs*2,
		shuffle=False,
		pin_memory=True,
		persistent_workers=True,
		num_workers=2
	)
	p = Pipeline(model, torch.nn.CrossEntropyLoss(), optim, scheduler_from_optim(optim))
	print('pre layer')
	p.train(trainloader, testloader, epochs, 5, True)

	print('layer 1')
	model.requires_grad_(False)
	model.layer1.requires_grad_()
	p.optimizer = sgd_from_model(model)
	p.scheduler = scheduler_from_optim(p.optimizer)
	p.train(trainloader, testloader, epochs, 5, True)
	
	print('layer 2')
	model.requires_grad_(False)
	model.layer2.requires_grad_()
	p.optimizer = sgd_from_model(model)
	p.scheduler = scheduler_from_optim(p.optimizer)
	p.train(trainloader, testloader, epochs, 5, True)

	
	print('layer 3')
	model.requires_grad_(False)
	model.layer3.requires_grad_()
	p.optimizer = sgd_from_model(model)
	p.scheduler = scheduler_from_optim(p.optimizer)
	p.train(trainloader, testloader, epochs, 5, True)

	print('layer 4')
	model.requires_grad_(False)
	model.layer4.requires_grad_()
	p.optimizer = sgd_from_model(model)
	p.scheduler = scheduler_from_optim(p.optimizer)
	p.train(trainloader, testloader, epochs, 5, True)

	print('cls')
	model.requires_grad_(False)
	model.fc.requires_grad_()
	p.optimizer = sgd_from_model(model)
	p.scheduler = scheduler_from_optim(p.optimizer)
	p.train(trainloader, testloader, epochs, 5, True)
	

if __name__ == '__main__':
	main()