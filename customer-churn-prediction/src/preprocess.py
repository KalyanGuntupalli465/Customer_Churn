import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('data/churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Filling missing values
print("Missing before fix:\n", df.isnull().sum())

df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print("\nMissing after fix:\n", df.isnull().sum())




#Dropping unnecessary colums

df.drop('customerID', axis=1, inplace=True)
print("Shape after dropping customerID:", df.shape)


#Encoding the target variable (churn)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("\nChurn value counts:\n", df['Churn'].value_counts())


#Encoding categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", cat_cols)

binary_cols = [col for col in cat_cols 
               if df[col].nunique() == 2]

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 
                            'Male': 1, 'Female': 0})

print("\nBinary columns encoded:", binary_cols)

multi_cols = [col for col in cat_cols 
              if df[col].nunique() > 2]

print("Multi-category columns:", multi_cols)

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print("\nShape after encoding:", df.shape)


#Creating meaningful features
df['ChargesPerMonth']=df['TotalCharges']/(df['tenure']+1)
df['IsNewCustomer']= (df['tenure']<6).astype(int)
df['IsHighSpender'] = (df['MonthlyCharges']> df['MonthlyCharges'].median()).astype(int)

print("New Featues are added")
print(df[['ChargesPerMonth', 'IsNewCustomer', 'IsHighSpender']].head())


#scaling numerical features

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesPerMonth']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nNumerical features scaled!")
print(df[num_cols].describe().round(2))


#Handling class imbalance

X=df.drop('Churn',axis=1)
y=df['Churn']

print("Before SMOTE:", y.value_counts().to_dict())

#Train test split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE:", pd.Series(y_train_sm).value_counts().to_dict())



#Saving preprocessed data
X_train_sm.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
pd.Series(y_train_sm).to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

joblib.dump(scaler, 'models/scaler.pkl')

print("\n✅ All processed data saved!")
print("X_train shape:", X_train_sm.shape)
print("X_test shape:", X_test.shape)