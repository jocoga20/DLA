import optuna
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import torch
import torchvision
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

from Pipeline import Pipeline

def feature_learning(train_features, train_labels, valid_features, valid_labels):
	from sklearn.svm import SVC
	model = SVC()

	model.fit(train_features, train_labels)
	train_pred = model.predict(train_features)
	valid_pred = model.predict(valid_features)

	from sklearn.metrics import accuracy_score
	print(f'Train {accuracy_score(train_labels, train_pred)}')
	print(f'Valid {accuracy_score(valid_labels, valid_pred)}')


def feature_extraction_batch(batch, model, tokenizer, device):
	inputs = tokenizer(
		batch['text'],
		padding=True,
		truncation=True,
		return_tensors='pt',
	).to(device)

	with torch.no_grad():
		outputs = model(**inputs)

	embeddings = outputs.last_hidden_state[:,0,:]

	return {'features': embeddings.cpu()}

def load_feature_dataset():
	slug = 'cornell-movie-review-data/rotten_tomatoes'
	model_slug = 'distilbert/distilbert-base-uncased'
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = AutoModel.from_pretrained(model_slug).to(device)
	tokenizer = AutoTokenizer.from_pretrained(model_slug)
	return load_dataset(path=slug, cache_dir='./mydatasets')\
	.map(
		lambda b: feature_extraction_batch(b, model, tokenizer, device),
		batched=True,
		batch_size=64
	)

def save_features_and_labels():
	ds_features = load_feature_dataset()
	train_features = np.array(ds_features['train']['features'])
	train_labels = np.array(ds_features['train']['label'])
	valid_features = np.array(ds_features['validation']['features'])
	valid_labels = np.array(ds_features['validation']['label'])

	np.save('train_features.npy', train_features)
	np.save('train_labels.npy', train_labels)
	np.save('valid_features.npy', valid_features)
	np.save('valid_labels.npy', valid_labels)

def load_features_and_labels():
	train_features = np.load('train_features.npy')
	train_labels = np.load('train_labels.npy')
	valid_features = np.load('valid_features.npy')
	valid_labels = np.load('valid_labels.npy')
	return train_features, train_labels, valid_features, valid_labels

def ft_obj_fun(trial: optuna.Trial, xtrain, ytrain, xvalid, yvalid):
	layer_options = [
		(64,),
		(128,),
		(256,),
		(512,),
		(128, 64),
		(256, 128)
	]
	model = MLPClassifier(
		hidden_layer_sizes=[256],
		activation='relu',
		solver='adam',
		learning_rate='constant',
		alpha=trial.suggest_float('alpha', 1e-3, 1e-2, log=True),
		learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-3, log=True),
		max_iter=100,
		beta_1=0,
		beta_2=0,
		n_iter_no_change=5,
		early_stopping=True
	)
	model.fit(xtrain, ytrain)
	valid_pred = model.predict(xvalid)
	return accuracy_score(yvalid, valid_pred)

def ft_batch_tokenize(batch, tokenizer):
	return tokenizer(
		batch['text'],
		padding='max_length',
		truncation=True,
		max_length=128
	)

def cp_met(eval_pred):
	preds, labels = eval_pred
	preds = np.argmax(preds, axis=1)
	return {
		'acc': accuracy_score(labels, preds)
	}

def bert_full_fine_tuning():
	from transformers import AutoModelForSequenceClassification
	from transformers.utils import logging
	logging.set_verbosity_error()
	slug = 'cornell-movie-review-data/rotten_tomatoes'
	model_slug = 'distilbert/distilbert-base-uncased'
	tokenizer = AutoTokenizer.from_pretrained(model_slug)
	model = AutoModelForSequenceClassification.from_pretrained(model_slug, num_labels=2)
	model.requires_grad_()
	model.distilbert.embeddings.requires_grad_(False)
	model.distilbert.transformer.layer[0].requires_grad_(False)
	tk_ds = load_dataset(slug, cache_dir='./mydatasets')\
		.map(lambda b: ft_batch_tokenize(b, tokenizer), batched=True, batch_size=128)\
		.remove_columns(['text'])\
		.rename_column('label', 'labels')
	tk_ds.set_format('torch')
	bs = 256
	# import wandb
	# wandb.init(project='DLA', name='bert ft')

	train_args = TrainingArguments(
		output_dir='./bert_results',
		eval_strategy='epoch',
		save_strategy='no',
		logging_strategy='no',
		learning_rate=5e-6,
		per_device_train_batch_size=bs,
		per_device_eval_batch_size=bs*2,
		num_train_epochs=20,
		weight_decay=1e-2,
		report_to='none',
		disable_tqdm=True
	)

	trainer = Trainer(
		model=model,
		args=train_args,
		train_dataset=tk_ds['train'],
		eval_dataset=tk_ds['validation'],
		compute_metrics=cp_met
	).train()


def hyperparameter_tuning():
	optuna.logging.set_verbosity(optuna.logging.WARN)
	study = optuna.create_study(direction='maximize')
	xtrain, ytrain, xvalid, yvalid = load_features_and_labels()
	study.optimize(lambda x: ft_obj_fun(x, xtrain, ytrain, xvalid, yvalid), n_trials=200, show_progress_bar=True, n_jobs=8)
	trials = sorted(
		study.trials,
		key=lambda t: t.value if t.value is not None else float('-inf'),
		reverse=True
	)

	with open('results_lab2.txt', 'w') as f:
		for trial in trials:
			f.write(
				f"# {trial.number} | "
				f"ACC: {trial.value*100:.4f} | "
				f"Params: {trial.params}\n"
			)

def clip_similarity():
	from transformers import CLIPModel, CLIPProcessor
	from PIL import Image

	model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
	processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
	file_names = ['prova.jpeg', 'prova2.jpeg', 'prova3.jpeg', '3.jpeg']
	image = [Image.open(fn) for fn in file_names]
	text = ['peach', 'hat', 'person']
	inputs = processor(text=text, images=image, return_tensors='pt', padding=True)
	outputs = model(**inputs)
	print(outputs.logits_per_image)

def get_oxford(split):
	return torchvision.datasets.OxfordIIITPet(root='./mydatasets', split=split, target_types='category', download=True)

def oxford_train():
	return get_oxford('trainval')

def oxford_test():
	return get_oxford('test')

def clip_classification():
	ds = oxford_test()
	print(ds[0])

if __name__ == '__main__':
	clip_classification()