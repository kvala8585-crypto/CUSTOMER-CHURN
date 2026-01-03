import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_path = r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_testing.csv"
test_path  = r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_training.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print(df_train.shape)
print(df_test.shape)
#data overview
print(df_train.head())
print(df_train.info())

#Target varible analysis to churn distribution to understand customer retention behavior.”
print(df_train["churn"].value_counts())
print(df_train["churn"].value_counts(normalize=True) * 100)
#Churn distribution plot 
# #0 = Not Churned
#1 = Churned
plt.figure()
sns.countplot(x="churn", data=df_train)
plt.title("Churn Distribution")
plt.show()
# Age vs churn analysis #Younger aur older customers ka churn behavior compare hota hai
plt.figure()
sns.boxplot(x="churn", y="age", data=df_train)
plt.title("Age vs Churn")
plt.show()
#Tenure vs Churn #Low tenure customers zyada churn karte hain
plt.figure()
sns.boxplot(x="churn", y="tenure", data=df_train)
plt.title("Tenure vs Churn")
plt.show()
#Catogirical Features analysis analyzed categorical features using frequency distribution.
cat_cols = df_train.select_dtypes(include="object").columns

for col in cat_cols:
    print("\n", col)
    print(df_train[col].value_counts().head())
#Churn by gender
plt.figure()
sns.countplot(x="gender", hue="churn", data=df_train)
plt.title("Gender vs Churn")
plt.show()
#Correlation Analysis (Numeric)  Highly correlated features help in model building
# Sirf numeric columns select karo
numeric_df = df_train.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#summary statics
df_train.describe()


#Performed univariate and bivariate analysis

#Analyzed churn distribution

#Studied relationship between churn and age, tenure, gender

#Used boxplots, countplots and correlation heatmap

#Prepared data for machine learning modeling

#target and features seprated #separated independent features and dependent target variable.”
X = df_train.drop("churn", axis=1)   # Features
y = df_train["churn"]                # Target

#Categorical column identify 
cat_cols = X.select_dtypes(include="object").columns
print(cat_cols)

#label encoding Male → 1, Female → 0
le = LabelEncoder()

for col in cat_cols:
    X[col] = le.fit_transform(X[col])
    #sab column numeric honi chahiye
    print(X.head())
    print(X.dtypes)
#train test split Meaning:80% → Training data 20% → Testing data
#stratify=y → churn ratio same rahe
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#shape check
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

#Cnverted categorical variables using Label Encoding

#Separated features and target variable

#Performed train-test split with 80–20 ratio

#Used stratified sampling to handle churn imbalance

#logistics regression 
from sklearn.linear_model import LogisticRegression

# Model create
model = LogisticRegression(max_iter=1000)

# Model train
model.fit(X_train, y_train)

print("✅ Logistic Regression model trained successfully")

#precision Model test data pe churn predict karega
y_pred = model.predict(X_test)

#Accuracy Score Kitni prediction sahi hui
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#Accuracy shows the percentage of correctly predicted churn customers.
#Confusion Matrix
#TP → sahi churn predict

#TN → sahi non-churn predict

#FP → galat churn predict

#FN → churn miss ho gaya
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
#classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
#classification report provide ,Precision ,Recall,F1-score,churn prediction

#Feature Importance (Logistic Regression me feature importance kaise?Coefficient ke through)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print(feature_importance)
#features importance plot
plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Important Features")
plt.show()
