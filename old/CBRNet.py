
class CBR(Module):
	def __init__(self, in_channels, out_channels, kernel):
		super().__init__()
		self.cn = torch.nn.Conv2d(
			in_channels,
			out_channels,
			kernel,
		)
		self.bn = torch.nn.BatchNorm2d(
			out_channels
		)
		self.r = torch.nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.cn(x)
		x = self.bn(x)
		return self.r(x)

class CBRNet(Module):
	def __init__(self, input_image_size:int, out_dim:int, chn_increse_factor:int, img_reduct_factor:int):
		super().__init__()
		self.cbrs = ModuleList()
		actual_dim = input_image_size
		actual_channels = 3
		while actual_dim >= img_reduct_factor:
			actual_dim = int(actual_dim / img_reduct_factor)
			self.cbrs.append(CBR(actual_channels, actual_channels * chn_increse_factor, actual_dim))
			actual_channels *= chn_increse_factor

		self.classifier = torch.nn.LazyLinear(out_dim)

	def forward(self, x):
		for layer in self.cbrs:
			x = layer(x)
		x = torch.flatten(x, 1)
		return self.classifier(x)

def count_parameters(model: Module):
	return sum(p.numel() for p in model.parameters())