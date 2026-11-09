from torchvision.models import resnet18
from Pipeline import Pipeline

def get_vision_model(num_classes: int):
	m = resnet18(weights=None, num_classes=num_classes)
	return m.cuda()

def load_model(checkpoint_file: str):
	return Pipeline(get_vision_model(10)).load_checkpoint(checkpoint_file).model