import wandb

class NoLogger:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc, tb): pass
    def log(self, step, data_dict): pass

class WandbLogger(NoLogger):
    def __init__(self, project: str, run_name: str = None, entity: str = None, config_dict: dict = None):
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self.config_dict = config_dict

    def __enter__(self):
        self.run = wandb.init(self.entity, self.project, name=self.run_name, config=self.config_dict)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.run.finish(exit_code=1 if exc_type else 0)
        return False
    
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