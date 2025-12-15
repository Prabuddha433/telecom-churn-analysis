# cell 1 - imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# cell 2 - load cleaned file if available else raw
try:
    df = pd.read_csv('../data/churn_clean.csv')
except:
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # quick cleaning inline
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', 0))
    df = df.replace({'No internet service': 'No', 'No phone service': 'No'})

# cell 2.1 - feature engineering for EDA

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 6, 12, 24, 60, 100],
    labels=["0-6", "6-12", "12-24", "24-60", "60+"]
)


# cell 3 - quick look
df.head(), df.shape

# cell 4 - churn rate
print("Churn distribution:")
print(df['Churn'].value_counts(normalize=True))
sns.countplot(data=df, x='Churn')

# cell 4.1 - churn by tenure group (UPGRADE 1 - STEP 3)

print("Churn rate by Tenure Group:")
print(
    df.groupby("TenureGroup")["Churn"]
      .value_counts(normalize=True)
)


# cell 5 - numeric distributions
numeric = df.select_dtypes(include=['int64','float64']).columns
df[numeric].hist(figsize=(12,8))

# cell 6 - tenure vs churn
sns.boxplot(data=df, x='Churn', y='tenure')

# cell 7 - correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric].corr(), annot=True, fmt=".2f")
