import torch
from tqdm import tqdm

class Run:
	def __init__(self, model, optimizer, loss):
		self.optimizer = optimizer
		self.loss = loss

		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
		print(self.device)
		self.model = model.to(self.device)
	
	def train(self, loader, progress_bar=False):
		self.model.train()

		epoch_loss = 0
		corrects = 0
		size = 0

		if progress_bar:
			loader = tqdm(loader)

		for x, y in loader:
			x = x.to(self.device)
			y = y.to(self.device)
			size += len(y)

			self.optimizer.zero_grad()

			logits = self.model(x)
			
			loss_tensor = self.loss(logits, y)
			epoch_loss += loss_tensor.item()

			loss_tensor.backward()

			self.optimizer.step()
			
			preds = logits.argmax(dim=1)
			corrects += (preds == y).sum().item()

		return epoch_loss / size, corrects / size

	def eval(self, loader, progress_bar=False):
		self.model.eval()

		epoch_loss = 0
		corrects = 0
		size = 0
		
		if progress_bar:
			loader = tqdm(loader)

		for x, y in loader:
			x = x.to(self.device)
			y = y.to(self.device)
			size += len(y)

			logits = self.model(x)
			
			loss_tensor = self.loss(logits, y)
			epoch_loss += loss_tensor.item()

			preds = logits.argmax(dim=1)
			
			corrects += (preds == y).sum().item()
	
		return epoch_loss / size, corrects / size