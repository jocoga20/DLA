from collections.abc import Callable
from typing import Literal

from torch.nn import Identity, Linear, Module

def replace_last_linear(model: Module, output_classes: int):
	for name, module in reversed(list(model.named_modules())):
		if isinstance(module, Linear):
			parts = name.split('.')

			parent = model
			for p in parts[:-1]:
				if p.isdigit():
					parent = parent[int(p)]
				else:
					parent = getattr(parent, p)

			child = parts[-1]

			new_layer = Identity() if output_classes <= 0 else Linear(module.in_features, output_classes)

			if child.isdigit():
				parent[int(child)] = new_layer
			else:
				setattr(parent, child, new_layer)

			return None

	raise RuntimeError("No Linear layer found")

def linear_probing(model: Module, output_classes: int):
	model.requires_grad_(False)
	replace_last_linear(model, output_classes)

def feature_extraction(model: Module):
	model.requires_grad_(False)
	replace_last_linear(model, 0)

def full_fine_tuning(model: Module, output_classes: int):
	model.requires_grad_()
	replace_last_linear(model, output_classes)