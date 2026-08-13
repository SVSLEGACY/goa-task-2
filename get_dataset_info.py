from datasets import load_dataset_builder

builder = load_dataset_builder("ai4bharat/MSMARCO-XI")
print(f"Dataset Name: {builder.info.builder_name}")
print(f"Description: {builder.info.description}")
print(f"Splits:")
for split_name, split_info in builder.info.splits.items():
    print(f" - {split_name}: {split_info.num_examples} examples (rows)")
