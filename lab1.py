import optuna
import torch
import torchvision
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from Pipeline import Pipeline
from torch import Tensor, inference_mode
from utils import get_train, get_test
from torch.nn import Identity
import torch.nn.functional as f
from get_models import *

def feature_extraction(model: Module, loader: DataLoader, save_path: str):
	logits = Pipeline(model).inference_logits(loader, True)
	torch.save(logits, save_path)

@inference_mode()
def feature_extraction_all_models(names: list[str], models: list[Module]):
	bs = 256
	nw = 0
	for name, (model, inf_transform) in zip(names, models):
		trainloader = DataLoader(
			dataset=get_train(inf_transform),
			batch_size=bs,
			shuffle=False,
			pin_memory=True,
			num_workers=nw
		)
		model = model.cuda()
		model.eval()
		model.fc = Identity()
		feature_extraction(model, trainloader, f'features.{name}.train.pt')
		del trainloader
		testloader = DataLoader(
			dataset=get_test(inf_transform),
			batch_size=bs,
			shuffle=False,
			pin_memory=True,
			num_workers=nw
		)
		feature_extraction(model, testloader, f'features.{name}.test.pt')
		del testloader

def get_loaders(xtrain, ytrain, train_batch_size, xtest, ytest, test_batch_size):
	trainloader = DataLoader(
		dataset=TensorDataset(xtrain, ytrain),
		batch_size=train_batch_size,
		shuffle=True,
		num_workers=5,
		pin_memory=True
	)
	testloader = DataLoader(
		dataset=TensorDataset(xtest, ytest),
		batch_size=test_batch_size,
		shuffle=False,
		num_workers=5,
		pin_memory=True
	)
	return trainloader, testloader

def linear_probing():
	names = ['resnet18', 'efficientnetb0', 'mobilenet', 'convnext']
	concept_dim = [512, 1280, 1024, 768]
	ytrain = torch.load('labels.train.pt').to(torch.long)
	ytest = torch.load('labels.test.pt').to(torch.long)
	for dim, name in zip(concept_dim, names):
		print(name)
		model = torch.nn.Linear(dim, 43, device='cuda')
		optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0)
		xtrain = torch.load(f'features.{name}.train.pt')
		xtest = torch.load(f'features.{name}.test.pt')
		trl, tel = get_loaders(xtrain, ytrain, 512, xtest, ytest, 1024)
		Pipeline(model, torch.nn.CrossEntropyLoss(reduction='sum'), optimizer)\
			.train(trl, tel, 100, patience=5)

def fine_tuning_loaders(train_batch, test_batch, transform, g):
	train = DataLoader(
		dataset=get_train(transform),
		batch_size=train_batch,
		shuffle=True,
		num_workers=4,
		pin_memory=True,
		persistent_workers=True,
		generator=g
	)

	test = DataLoader(
		dataset=get_test(transform),
		batch_size=test_batch,
		shuffle=False,
		num_workers=2,
		pin_memory=True,
		persistent_workers=True
	)
	return train, test

def set_seed(seed=42):

#	random.seed(seed)
#	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	g = torch.Generator()
	g.manual_seed(seed)
	return g

def lock_mobilenet_layers(model: Module, layer_indexes):
	for i in layer_indexes:
		model.features[i].requires_grad_(False)

def prepare_mobilenet(model: Module, lock_level: int):
	model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 43)
	model.requires_grad_()
	
	if lock_level > 0:
		lock_mobilenet_layers(model, range(4))
		if lock_level > 1:
			lock_mobilenet_layers(model, range(4, 8))
			if lock_level > 2:
				lock_mobilenet_layers(model, range(8, 13))
	return model.cuda()

def prepare_resnet18(model: Module):
	model.fc = torch.nn.Linear(model.fc.in_features, 43)
	model.requires_grad_(False)
	model.fc.requires_grad_()
	return model.cuda()

def prepare_resnet34(model: Module):
	model.fc = torch.nn.Linear(model.fc.in_features, 43)
	model.requires_grad_(False)
	model.fc.requires_grad_()
	return model.cuda()

def prepare_efficientnet(model: Module):
	model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 43)
	model.requires_grad_(False)
	model.classifier.requires_grad_()
	return model.cuda()

def fine_tuning(trial: optuna.Trial):
	g = set_seed(42)
	lock_level = trial.suggest_categorical('lock', [0, 1, 2, 3])
	model, inf_transform = get_mobilenet()
	model = prepare_mobilenet(model, lock_level)

	lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
	wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
	optim = torch.optim.SGD(params=[p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
	epochs = 20
	max_patience = 5
	patience = 0
	best_acc = 0
	trainloader, testloader = fine_tuning_loaders(256, 512, inf_transform, g)
	p = Pipeline(model, torch.nn.CrossEntropyLoss(), optim)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

	for e in range(epochs):
		p.train_epoch(trainloader)
		_, test_acc = p.eval_epoch(testloader)
		if test_acc > best_acc:
			best_acc = test_acc
			patience = 0
		else:
			patience += 1
			if patience >= max_patience:
				return best_acc
		scheduler.step()
	del model, trainloader, testloader
	return best_acc

def query(first_features: Tensor, second_features: Tensor):
	first_features = f.normalize(first_features, p=2, dim=1)
	second_features = f.normalize(second_features, p=2, dim=1)
	return first_features @ second_features.T

def compute_similarities(first_file: str, second_file: str):
	return query(torch.load(first_file), torch.load(second_file))

def rank_by_index(query_matrix: Tensor, k: int):
	return query_matrix.topk(k, dim=1).indices

def hyperparameter_search():
	optuna.logging.set_verbosity(optuna.logging.WARN)
	study = optuna.create_study(direction='maximize')
	study.optimize(fine_tuning, n_trials=20, show_progress_bar=True, n_jobs=2)
	trials = sorted(
		study.trials,
		key=lambda t: t.value if t.value is not None else float('-inf'),
		reverse=True
	)

	with open('results.txt', 'w') as f:
		for trial in trials:
			f.write(
				f"# {trial.number} | "
				f"ACC: {trial.value*100:.4f} | "
				f"Params: {trial.params}\n"
			)

def rank_matrix_to_class(ranks, labels):
	return labels[ranks]

def evaluate_retrieval(topk_results: Tensor, true_labels: Tensor):
	# topk_results = N_test x k
	# true_labels = N_test

	corrects = (topk_results == true_labels[:,None]).any(dim=1).sum().item()
	print(f'Accuracy: {corrects / len(true_labels)}')
	
def main_ranking(name, k):
	print('Similarity scores')
	q = compute_similarities(name_to_test_file(name), name_to_train_file(name)) # n_test x n_train
	topk_rank = rank_by_index(q, k)
	train_labels = torch.load('labels.train.pt')
	test_labels = torch.load('labels.test.pt')
	topk_classes = rank_matrix_to_class(topk_rank, train_labels)
	
	evaluate_retrieval(topk_classes, test_labels)

def compare_rankings(names, k):
	for name in names:
		print(name)
		main_ranking(name, k)

def centroid_by_index(matrix: Tensor, indexes: Tensor):
	return matrix[indexes].mean(dim=0)

def name_to_train_file(name):
	return f'features.{name}.train.pt'

def name_to_test_file(name):
	return f'features.{name}.test.pt'

def element_to_class(instance_matrix: Tensor, centroid_matrix: Tensor):
	distances = torch.cdist(instance_matrix, centroid_matrix)
	return distances.argmin(dim=1)

def nearest_mean_classifier(name):
	x_train = torch.load(name_to_train_file(name))
	x_test = torch.load(name_to_test_file(name))
	train_labels = torch.load('labels.train.pt')
	test_labels = torch.load('labels.test.pt')
	num_classes = 43
	centroid_matrix = torch.vstack([centroid_by_index(x_train, train_labels == i) for i in range(num_classes)])
	centroid_matrix = torch.nn.functional.normalize(centroid_matrix)
	ypred = element_to_class(x_test, centroid_matrix)
	accuracy = (ypred == test_labels).sum().item()
	return accuracy / len(test_labels)

if __name__ == '__main__':
	hyperparameter_search()