from matplotlib import pyplot as plt
import numpy as np
import torch


@torch.inference_mode()
def to_logits(load_model_path: str):
	model = get_vision_model(num_classes=10).cuda()
	model.load_state_dict(torch.load(load_model_path)['model'])
	model.eval()
	test_size = 10_000

	id_loader = DataLoader(
		dataset=get_cifar10(train=False, transform=test_transform),
		batch_size=1024,
		shuffle=False,
		num_workers=4,
		pin_memory=True
	)
	torch.save(get_logits(model, id_loader).cpu(), 'id_logits.pt')

	ood_loader = DataLoader(
		dataset=get_svhn(size=test_size, transform=test_transform),
		batch_size=1024,
		shuffle=False,
		num_workers=4,
		pin_memory=True
	)
	torch.save(get_logits(model, ood_loader).cpu(), 'ood_logits.pt')

def show_scores(id_scores: np.array, ood_scores: np.array):
	plt.figure(figsize=(8, 5))

	plt.hist(id_scores, bins=50, alpha=0.5, density=True, label="ID")
	plt.hist(ood_scores, bins=50, alpha=0.5, density=True, label="OOD")

	plt.xlabel("Confidence score")
	plt.ylabel("Density")
	plt.legend()
	plt.grid(True)

	plt.show()