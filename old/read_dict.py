import json

fname = '/home/jonathan/Desktop/DLA/bert_results/checkpoint-1340/trainer_state.json'
with open(fname, 'r') as f:
    data = json.load(f)
data = data['log_history']

eval_accs = [d['eval_accuracy'] for d in data if 'eval_accuracy' in d.keys()]
print(max(eval_accs))