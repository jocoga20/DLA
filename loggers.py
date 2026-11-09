import wandb

class NoLogger:
    def log(self, step, data_dict): pass
    def end_run(self): pass

class WandbLogger(NoLogger):
    def __init__(self, project: str, run_name:str = None, entity:str = None, config_dict: dict = None):
        self.run = wandb.init(project=project, name=run_name, entity=entity, config=config_dict)
    
    def log(self, step, data_dict):
        self.run.log(data=data_dict, step=step)
    
    def end_run(self):
        self.run.finish()

class FormatLogger(NoLogger):
    def format_string(self, step: int, data_dict: dict):
        return f'{step} {data_dict}'

class PrintLogger(FormatLogger):
    def log(self, step, data_dict: dict):
        print(super().format_string(step, data_dict))

class FileLogger(FormatLogger):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def log(self, step: int, data_dict: dict):
        with open(self.file_path, 'a') as file:
            file.write(super().format_string(step, data_dict) + '\n')