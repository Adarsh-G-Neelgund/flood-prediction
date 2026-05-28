import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('flood_risk_dataset_india.csv')


numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df.drop_duplicates(inplace=True)


features_to_check = numeric_cols.drop('Flood_Occurred', errors='ignore')
Q1 = df[features_to_check].quantile(0.25)
Q3 = df[features_to_check].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[features_to_check] < (Q1 - 3.0 * IQR)) | (df[features_to_check] > (Q3 + 3.0 * IQR))).any(axis=1)]


X = df.drop('Flood_Occurred', axis=1)
y = df['Flood_Occurred']


X_encoded = pd.get_dummies(X, drop_first=True)

dummy_columns = list(X_encoded.columns)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)


pca = PCA(n_components=0.95)
X_extracted = pca.fit_transform(X_selected)


model = LogisticRegression(random_state=42)
model.fit(X_extracted, y)


joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(dummy_columns, 'dummy_columns.pkl')

print("All ML pipeline components saved successfully without errors!")