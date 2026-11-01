from dataclasses import dataclass


@dataclass
class LogData:
    epoch: int
    train_loss: float
    eval_loss: float
    train_acc: float
    eval_acc: float