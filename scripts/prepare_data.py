from src.data.parser import process_dataset

df = process_dataset("data/raw/cairo.txt")
df.to_csv("data/processed/cairo.csv", index=False)
