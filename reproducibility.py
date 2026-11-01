import random
import numpy as np
import torch

def set_seed(seed: int):

	random.seed(seed)
	np.random.seed(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	deterministic = True
	torch.backends.cudnn.deterministic = deterministic
	torch.backends.cudnn.benchmark = not deterministic

	torch.use_deterministic_algorithms(deterministic)
	g = torch.Generator()
	g.manual_seed(seed)
	return g