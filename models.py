from typing import Literal

import torch

def freeze_model(model: torch.nn.Module):
	for p in model.parameters():
		p.requires_grad_(False)         # more correct than raw assignment

def replace_last_linear(model, last_layer):
	for name, module in reversed(list(model.named_modules())):
		if isinstance(module, torch.nn.Linear):
			parts = name.split('.')

			parent = model
			for p in parts[:-1]:
				if p.isdigit():
					parent = parent[int(p)]
				else:
					parent = getattr(parent, p)

			child = parts[-1]

			new_layer = last_layer(module.in_features)

			if child.isdigit():
				parent[int(child)] = new_layer
			else:
				setattr(parent, child, new_layer)
		return

	raise RuntimeError("No Linear layer found")

def rework_model(model, last_layer: Literal['linear', 'identity'] = 'linear', output_classes: int = None, do_freeze_backbone: bool = True):
	if do_freeze_backbone:
		freeze_model(model)
	if last_layer == 'linear':
		replace_last_linear(model, lambda x: torch.nn.Linear(x, output_classes))
	else:
		replace_last_linear(model, lambda x: torch.nn.Identity())