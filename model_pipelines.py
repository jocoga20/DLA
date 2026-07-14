from torch.nn import Identity, Linear, Module

def freeze_all(model: Module):
	for p in model.parameters():
		p.requires_grad_(False)         # more correct than raw assignment

def replace_last_linear(model, last_layer):
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

			new_layer = last_layer(module.in_features)

			if child.isdigit():
				parent[int(child)] = new_layer
			else:
				setattr(parent, child, new_layer)
		return None

	raise RuntimeError("No Linear layer found")

def linear_probing(model: Module, output_classes: int):
	freeze_all(model)
	replace_last_linear(model, lambda x: Linear(x, output_classes))

def feature_extraction(model: Module):
	freeze_all(model)
	replace_last_linear(model, lambda x: Identity())

def full_fine_tuning(model: Module, output_classes: int):
	replace_last_linear(model, lambda x: Linear(x, output_classes))