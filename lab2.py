import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np

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

def main():
    train_features, train_labels, valid_features, valid_labels = load_features_and_labels()
    feature_learning(train_features, train_labels, valid_features, valid_labels)

main()