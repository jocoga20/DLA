import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from FeatureDataset import FeatureDataset

def replace_comma(hl):
    return str(hl).replace(',', '_')

def feature_learning(feature_datasets: list[FeatureDataset], rng_seed, result_csv_path = None, p_bar = False):
    mlp_hidden_layers = [[], [2], [128, 32]]
    knn_neighbors = [3,5,7,9]

    models = [(f'{i}-NN', KNeighborsClassifier(n_neighbors=i)) for i in knn_neighbors]\
          + [(f'MLP {replace_comma(hl)}', MLPClassifier(hidden_layer_sizes=hl, random_state=rng_seed, max_iter=200, early_stopping=True)) for hl in mlp_hidden_layers]

    data = []

    y_train = torch.load(feature_datasets[0].label_path('train'))
    y_test = torch.load(feature_datasets[0].label_path('test'))
    
    if p_bar:
        feature_datasets = tqdm(feature_datasets)

    scaler = StandardScaler()

    for fd in tqdm(feature_datasets):
        train_features = scaler.fit_transform(torch.load(fd.feature_path('train')).numpy())

        for model_name, model in tqdm(models):
            model.fit(train_features, y_train)
            ypred_train = model.predict(train_features)
            ypred_test = model.predict(torch.load(fd.feature_path('test')).numpy())
            data.append({
                'Backbone': fd.backbone_name,
                'Model': model_name,
                'BA_Train': balanced_accuracy_score(y_train, ypred_train),
                'BA_Test': balanced_accuracy_score(y_test, ypred_test),
                'ACC_Train': accuracy_score(y_train, ypred_train),
                'ACC_Test': accuracy_score(y_test, ypred_test)
            })

    data = pd.DataFrame(data)
    if result_csv_path:
        data.to_csv(result_csv_path, index=False)
    else:
        return data