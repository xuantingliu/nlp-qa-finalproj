from utils import load_dataset

meta, samples = load_dataset("datasets/squad_train.jsonl.gz")
print(meta)
print()
print(len(samples))
print(samples[:1])
