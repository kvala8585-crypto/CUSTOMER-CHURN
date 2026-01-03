import pandas as pd

train_csv_path = r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_train.csv"
test_csv_path  = r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_test.csv"

train_df = pd.read_csv(r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_train.csv")
test_df  = pd.read_csv(r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_test.csv"
)
print(train_df.shape)
print(test_df.shape)
