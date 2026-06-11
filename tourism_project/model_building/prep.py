# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/jayaprakashp1/tourism-package-prediction/tourism.csv"
bank_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Remove the unnecessary columns - we can remove customer Id as it contains unique values for all rows.
bank_dataset = bank_dataset.drop('CustomerID', axis=1)

# Define the target variable for the classification task
target = 'ProdTaken'

# Data correction
# The column Gender has a spelling mistake for few rows, it has 'Fe male' instead of Female, so we need to correct it.
bank_dataset['Gender'] = bank_dataset['Gender'].replace('Fe Male', 'Female')

#Encode the categorical variable Gender
bank_dataset['Gender'] = bank_dataset['Gender'].map({
    'Male': 0,
    'Female': 1
})

# Define predictor matrix (X)
X = bank_dataset.drop('ProdTaken', axis=1)

# Define target variable
y = bank_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="jayaprakashp1/tourism-package-prediction",
        repo_type="dataset",
    )
