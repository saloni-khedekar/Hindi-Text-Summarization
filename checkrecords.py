import pandas as pd

# Load the train dataset
train_df = pd.read_csv("F:/VIT 3rd YEAR/6th Semester/EDI/train.csv")

# Load the test dataset
test_df = pd.read_csv("F:/VIT 3rd YEAR/6th Semester/EDI/test.csv")

# Get the number of records in the train dataset
num_records_train = len(train_df)
print("Number of records in train dataset:", num_records_train)

# Get the number of records in the test dataset
num_records_test = len(test_df)
print("Number of records in test dataset:", num_records_test)
