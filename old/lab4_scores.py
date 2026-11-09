import torch


def msp_score(logits: torch.Tensor):
	return logits.softmax(dim=1).amax(dim=1)      # .amax = max().values

def energy_score(logits: torch.Tensor, temperature: float = 1):
	return -temperature * logits.exp().sum(dim=1).log()