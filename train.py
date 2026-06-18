import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("📥 Step 1: Generating and preparing synthetic flood risk dataset...")
np.random.seed(42)
n_samples = 5000

# Generating 13-feature climate and geographic metrics matching your layout schema
data = {
    'Latitude': np.random.uniform(5.0, 40.0, size=n_samples),
    'Longitude': np.random.uniform(60.0, 100.0, size=n_samples),
    'Rainfall_mm': np.random.gamma(shape=3, scale=50, size=n_samples),
    'Temperature_C': np.random.normal(loc=25, scale=8, size=n_samples),
    'Humidity': np.random.randint(20, 100, size=n_samples),
    'River_Discharge_m3_s': np.random.normal(loc=2000, scale=800, size=n_samples),
    'Water_Level_m': np.random.normal(loc=4.5, scale=2.5, size=n_samples),
    'Elevation_m': np.random.exponential(scale=300, size=n_samples),
    'Land_Cover': np.random.choice(['Water Body', 'Forest', 'Agricultural', 'Desert', 'Urban'], size=n_samples),
    'Soil_Type': np.random.choice(['Clay', 'Peat', 'Loam', 'Sandy', 'Silt'], size=n_samples),
    'Population_Density': np.random.uniform(500, 15000, size=n_samples),
    'Infrastructure': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
    'Historical_Floods': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
}
df = pd.DataFrame(data)

# Hidden objective logic to simulate realistic ground truth risk labels
risk_score = (df['Rainfall_mm'] * 0.3) + (df['Water_Level_m'] * 2.0) - (df['Elevation_m'] * 0.02) + (df['River_Discharge_m3_s'] * 0.002)
risk_score += np.where(df['Infrastructure'] == 0, 40, 0)
risk_score += np.where(df['Historical_Floods'] == 1, 30, 0)
df['Flood_Occurred'] = np.where(risk_score > np.percentile(risk_score, 75), 1, 0)

# Outlier Filtering Strategy: 3.0x IQR cleaning rules
numerical_features = ['Rainfall_mm', 'Temperature_C', 'Humidity', 'River_Discharge_m3_s', 'Water_Level_m', 'Elevation_m', 'Population_Density']
for col in numerical_features:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 3.0 * IQR)) | (df[col] > (Q3 + 3.0 * IQR)))]

X = df.drop('Flood_Occurred', axis=1)
y = df['Flood_Occurred']

# Create dummy features using the exact format expected by the app pipeline
X_encoded = pd.get_dummies(X)

# Freeze layout blueprint tracking array
dummy_columns = list(X_encoded.columns)
with open('dummy_columns.pkl', 'wb') as f:
    pickle.dump(dummy_columns, f)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42)

print("⚙️ Step 2: Training Model A (Optimized Logistic Regression Pipeline)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k='all')
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

pca = PCA(n_components=0.95, random_state=42)
X_train_extracted = pca.fit_transform(X_train_selected)
X_test_extracted = pca.transform(X_test_selected)

model_lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
model_lr.fit(X_train_extracted, y_train)

y_pred_lr = model_lr.predict(X_test_extracted)
acc_lr, prec_lr, rec_lr = accuracy_score(y_test, y_pred_lr)*100, precision_score(y_test, y_pred_lr)*100, recall_score(y_test, y_pred_lr)*100

with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('selector.pkl', 'wb') as f: pickle.dump(selector, f)
with open('pca.pkl', 'wb') as f: pickle.dump(pca, f)
with open('logistic_regression_model.pkl', 'wb') as f: pickle.dump(model_lr, f)

print("📉 Step 3: Training Model B (Unoptimized Naïve Bayes Baseline)...")
model_base = GaussianNB()
model_base.fit(X_train, y_train)

with open('baseline_model.pkl', 'wb') as f: 
    pickle.dump(model_base, f)

y_pred_base = model_base.predict(X_test)
acc_base, prec_base, rec_base = accuracy_score(y_test, y_pred_base)*100, precision_score(y_test, y_pred_base)*100, recall_score(y_test, y_pred_base)*100

print("\n" + "="*60 + "\n🏁 MODEL BENCHMARK REPORT\n" + "="*60)
print(f"🏆 MODEL A (Logistic Regression): Acc: {acc_lr:.2f}% | Prec: {prec_lr:.2f}% | Rec: {rec_lr:.2f}%")
print(f"📉 MODEL B (Naïve Bayes):         Acc: {acc_base:.2f}% | Prec: {prec_base:.2f}% | Rec: {rec_base:.2f}%")
print("="*60 + "\n🚀 Success! New model files are ready to deploy.")