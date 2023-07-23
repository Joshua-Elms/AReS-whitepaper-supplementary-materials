from pathlib import Path
import pandas as pd


def main(dataset_path, subset_path, shape):
    pd.read_csv(dataset_path, nrows=shape[0]).to_csv(subset_path, index=False)
    print(f"Subset of dataset is available at {subset_path}")
    

if __name__ == "__main__":
    main(
        dataset_path = Path("datasets/test_datasets/concrete.csv"),
        subset_path = Path("datasets/test_datasets/concrete_subset.csv"),
        shape = (9999, 99),
    )