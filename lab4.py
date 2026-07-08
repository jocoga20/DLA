import torchvision


ds_folder = './mydatasets'
train_cf10 = torchvision.datasets.CIFAR10(ds_folder, train=True, download=True)
test_cf10 = torchvision.datasets.CIFAR10(ds_folder, train=False, download=True)
train_cf100 = torchvision.datasets.CIFAR100(ds_folder, train=True, download=True)
test_cf100 = torchvision.datasets.CIFAR10(ds_folder, train=False, download=True)