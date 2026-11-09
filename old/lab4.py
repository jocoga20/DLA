from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
import torchvision
from tqdm import tqdm

from Pipeline import Pipeline
from WarmupCosineLR import WarmupCosineLR
from adversarial import test_epoch_adv, train_epoch_adv
from lab4_datasets import get_cifar10, get_svhn
from lab4_transforms import train_transform, test_transform
from torch.utils.data import DataLoader
from torch.nn import ModuleList, Module

from model_pipelines import replace_last_linear

def feature_extraction(model: Module, loader: DataLoader, xsave: str, ysave: str):
	logits = Pipeline(model).inference_logits(loader, True)
	torch.save(logits, xsave)
	ys = []
	for _, y in loader:
		ys.append(y.cpu())
	ys = torch.cat(ys)
	torch.save(ys, ysave)

def main_adversarial_training():
	seed = 0
	g = set_seed(seed)
	total_epochs = 100
	torch.set_float32_matmul_precision('high')
	model = get_vision_model(num_classes=10)
	optim = torch.optim.SGD(
		params=[p for p in model.parameters() if p.requires_grad],
		lr=1e-3,
		momentum=0.5,
		nesterov=True,
		weight_decay=1e-3
	)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optim,
		T_0=20,
		T_mult=2,
		eta_min=1e-5
	)

	p = Pipeline(model, torch.nn.CrossEntropyLoss(reduction='sum'), optim, scheduler)
	trainloader=DataLoader(
		dataset=get_cifar10(True, train_transform),
		batch_size=512,
		shuffle=True,
		pin_memory=True,
		num_workers=12,
		persistent_workers=True,
		generator=g,
		in_order=False
	)

	testloader=DataLoader(
		dataset=get_cifar10(False, test_transform),
		batch_size=512,
		shuffle=False,
		pin_memory=True,
		in_order=False
	)

	loss_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

	for e in range(total_epochs):
		model.train()
		train_loss, train_acc = train_epoch_adv(model, trainloader, optim, loss_criterion, epsilon=4/256, progress_bar=True)
		scheduler.step()

		model.eval()
		
		test_loss, test_acc = p.eval_epoch(testloader, True)
		test_adv_loss, test_adv_acc = test_epoch_adv(model, testloader, loss_criterion, epsilon=4 / 256, progress_bar=True)
		print(f'[{e}] L: {train_loss:.4f} {test_loss:.4f} {test_adv_loss:.4f} | A (%): {train_acc*100:.2f} {test_acc*100:.2f} {test_adv_acc*100:.2f}')
	

def auroc_score(id_scores, ood_scores):
	id = id_scores.numpy()
	ood = ood_scores.numpy()
	ytrue = np.concat([np.ones_like(id), np.zeros_like(ood)])
	ypred = np.concat([id, ood])
	return sklearn.metrics.roc_auc_score(ytrue, ypred)

def aupr_score(id_scores, ood_scores):
	id = id_scores.numpy()
	ood = ood_scores.numpy()
	ytrue = np.concat([np.ones_like(id), np.zeros_like(ood)])
	ypred = np.concat([id, ood])
	return sklearn.metrics.average_precision_score(ytrue, ypred)

def logits_to_score(id_logits_path: str, ood_logits_path: str):
	id_logits = torch.load(id_logits_path)
	ood_logits = torch.load(ood_logits_path)
	print(auroc_score(id_logits, ood_logits),aupr_score(id_logits, ood_logits))

def to_image(x, name):
	from PIL import Image
	from torch.nn import functional as f

	img = x[0].detach().cpu().clamp(0, 1)
	img = (img * 255).byte()
	img = img.permute(1, 2, 0)
	pil_img = Image.fromarray(img.numpy())
	print(f'Saving {name}')
	pil_img.save(name)


if __name__ == '__main__':
	train_model()