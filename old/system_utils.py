import os

def get_max_workers():
	"""
	Max number of workers for DataLoader
	"""
	return len(os.sched_getaffinity(0))