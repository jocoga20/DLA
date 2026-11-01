from dataclasses import dataclass

@dataclass
class TrainConfig:
	epochs: int
	optimizer_config: dict
	rng_seed: int
	patience: int = None
	scheduler_config: dict | str = 'none'