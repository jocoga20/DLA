import wandb

class Logger:
	def log(self, step, data_dict): pass
	def end_run(self): pass

class WandbLogger(Logger):
	def __init__(self, project, run_name=None, api_key='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'):
		wandb.login(api_key, force=True)
		self.run = wandb.init(project=project, name=run_name)
	
	def log(self, step, data_dict):
		self.run.log(data=data_dict, step=step)
	
	def end_run(self):
		self.run.finish()

class PrintLogger(Logger):
	def log(self, step, data_dict):
		print(f'[{step}] {data_dict}\n')

class FileLogger(Logger):
	def __init__(self, file_path):
		super().__init__()
		self.file_path = file_path
	
	def log(self, step, data_dict):
		with open(self.file_path, 'a') as file:
			file.write(f'[{step}] {data_dict}\n')