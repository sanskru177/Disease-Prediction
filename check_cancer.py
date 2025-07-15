import pandas as pd
import os

# Path to dataset
cancer_path = os.path.join("..", "datasets", "breast_cancer.csv")
cancer = pd.read_csv(cancer_path)

# Check unique values in diagnosis column
print("Diagnosis Value Counts:")
print(cancer["diagnosis"].value_counts())
