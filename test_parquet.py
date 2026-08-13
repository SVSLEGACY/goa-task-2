import pandas as pd
from huggingface_hub import hf_hub_download

print("Downloading one parquet file...")
file_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="default/train/0000.parquet", repo_type="dataset")
print(f"Downloaded to: {file_path}")

print("Reading with pandas...")
df = pd.read_parquet(file_path)
print(f"Successfully loaded {len(df)} rows!")
print("Columns:", df.columns.tolist())
print(df.head(1))
