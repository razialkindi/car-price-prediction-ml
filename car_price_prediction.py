# -*- coding: utf-8 -*-
"""car_price_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EKljfYX5lojwXfHG78rh-9kL6SjG2uz8

# PREDIKSI HARGA MOBIL BEKAS - MACHINE LEARNING TERAPAN

## 1. Import library yang dibutuhkan
"""

import pandas as pd                                                             # Untuk manipulasi dan analisis data
import numpy as np                                                              # Untuk operasi numerik
import matplotlib.pyplot as plt                                                 # Untuk visualisasi data
import seaborn as sns                                                           # Untuk visualisasi data yang lebih canggih
from sklearn.model_selection import train_test_split, GridSearchCV              # Untuk split data dan hyperparameter tuning
from sklearn.preprocessing import StandardScaler, LabelEncoder                  # Untuk normalisasi fitur dan encoding
from sklearn.linear_model import LinearRegression                               # Model regresi linear
from sklearn.tree import DecisionTreeRegressor                                  # Model decision tree
from sklearn.ensemble import RandomForestRegressor                              # Model random forest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   # Metrik evaluasi
import warnings
warnings.filterwarnings('ignore')                                               # Mematikan peringatan untuk tampilan yang lebih bersih

"""## 2. Load Dataset dan Data Understanding

"""

print("Loading dataset...")
# Dataset Car Price Prediction dari Kaggle
url = "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"
df = pd.read_csv("/content/car_data.csv")

"""### Jumlah Data (Baris dan Kolom)

"""

print(f"Shape: {df.shape}")

"""Dataset memiliki 301 baris dan 9 kolom. Ini merupakan dataset yang cukup untuk analisis harga mobil bekas.

### Tipe Data
"""

print("\nData Types:")
print(df.dtypes)

"""*   Kita perlu memahami tipe data dari setiap kolom untuk menentukan metode preprocessing yang sesuai.
*   Tipe data numerik akan memerlukan normalisasi, sedangkan tipe data kategorikal memerlukan encoding.

### Statistik Deskriptif
"""

print("\n=== Statistical Summary ===")
print(df.describe())

"""*   Statistik deskriptif membantu kita memahami distribusi dan rentang nilai dari setiap fitur.
*   Informasi ini penting untuk mendeteksi outlier dan menentukan metode normalisasi yang tepat.

### Kondisi Data (Missing Value, Duplikat, Outlier)
"""

print("\n=== Check Missing Values ===")
print(df.isnull().sum())

print("\n=== Check Duplicate Values ===")
print(f"Duplicate rows: {df.duplicated().sum()}")

"""*   Berdasarkan hasil pengecekan, kita akan melihat apakah ada nilai yang hilang atau data duplikat yang perlu ditangani pada tahap preprocessing.

### Uraian Fitur pada Data
"""

print("\n=== Dataset Info ===")
print(df.info())

"""Dataset ini berisi data penjualan mobil bekas yang mencakup fitur-fitur berikut:

- Car_Name: Nama/model mobil
- Year: Tahun pembuatan mobil
- Selling_Price: Harga jual (target)
- Present_Price: Harga saat ini di showroom
- Kms_Driven: Jarak tempuh dalam kilometer
- Fuel_Type: Jenis bahan bakar (Petrol/Diesel/CNG)
- Seller_Type: Jenis penjual (Dealer/Individual)
- Transmission: Jenis transmisi (Manual/Automatic)
- Owner: Jumlah pemilik sebelumnya

## 3. Exploratory Data Analysis (EDA)

### Visualisasi Distribusi Harga Mobil
"""

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(df['Selling_Price'], kde=True, bins=30)
plt.title('Distribusi Harga Jual Mobil')
plt.xlabel('Harga Jual (Lakh)')

"""- Dari histogram, kita dapat melihat distribusi harga jual mobil bekas.
- Mayoritas mobil memiliki harga di bawah 10 lakh, dengan beberapa mobil mewah yang memiliki harga tinggi.

### Hubungan antara Tahun dan Harga
"""

plt.subplot(2, 3, 2)
sns.scatterplot(x='Year', y='Selling_Price', data=df)
plt.title('Harga vs. Tahun Pembuatan')
plt.xlabel('Tahun')
plt.ylabel('Harga Jual (Lakh)')

"""- Scatter plot menunjukkan korelasi positif antara tahun pembuatan dan harga jual.
- Mobil yang lebih baru cenderung memiliki harga yang lebih tinggi.

### Hubungan antara Kilometer dan Harga
"""

plt.subplot(2, 3, 3)
sns.scatterplot(x='Kms_Driven', y='Selling_Price', data=df)
plt.title('Harga vs. Kilometer Tempuh')
plt.xlabel('Kilometer Tempuh')
plt.ylabel('Harga Jual (Lakh)')

"""- Scatter plot menunjukkan korelasi negatif antara kilometer tempuh dan harga jual.
- Mobil dengan kilometer lebih rendah cenderung memiliki harga yang lebih tinggi.

### Distribusi berdasarkan Jenis Bahan Bakar
"""

plt.subplot(2, 3, 4)
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df)
plt.title('Harga berdasarkan Jenis Bahan Bakar')
plt.xlabel('Jenis Bahan Bakar')
plt.ylabel('Harga Jual (Lakh)')

"""### Distribusi berdasarkan Jenis Transmisi"""

plt.subplot(2, 3, 5)
sns.boxplot(x='Transmission', y='Selling_Price', data=df)
plt.title('Harga berdasarkan Jenis Transmisi')
plt.xlabel('Jenis Transmisi')
plt.ylabel('Harga Jual (Lakh)')

"""### Distribusi berdasarkan Jenis Penjual"""

plt.subplot(2, 3, 6)
sns.boxplot(x='Seller_Type', y='Selling_Price', data=df)
plt.title('Harga berdasarkan Jenis Penjual')
plt.xlabel('Jenis Penjual')
plt.ylabel('Harga Jual (Lakh)')

plt.tight_layout()
plt.show()

"""### Matriks Korelasi"""

plt.figure(figsize=(10, 8))
# Ambil hanya kolom numerik
numerical_df = df.select_dtypes(include=['number'])

# Hitung korelasi hanya untuk kolom numerik
correlation = numerical_df.corr()

# Buat mask segitiga atas
mask = np.triu(np.ones_like(correlation, dtype=bool))

# Plot heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
plt.title('Correlation Heatmap')
plt.show()

"""Dari heatmap korelasi, fitur-fitur yang memiliki korelasi tinggi dengan harga jual adalah:
- Present_Price (harga showroom saat ini)
- Year (tahun pembuatan)
- Kms_Driven (kilometer tempuh) - korelasi negatif

### Analisis Categorical Features
"""

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df['Fuel_Type'].value_counts().plot(kind='bar')
plt.title('Distribusi Jenis Bahan Bakar')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
df['Seller_Type'].value_counts().plot(kind='bar')
plt.title('Distribusi Jenis Penjual')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
df['Transmission'].value_counts().plot(kind='bar')
plt.title('Distribusi Jenis Transmisi')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

"""## 4. Data Preparation

### 4.1 Handling Missing Values
"""

print("Checking and handling missing values...")
print(df.isnull().sum())

"""- Jika ada missing values, kita akan mengatasinya sesuai dengan karakteristik masing-masing kolom

### 4.2 Feature Engineering
"""

print("Creating new features...")

# Age of the car (umur mobil)
df['Car_Age'] = 2024 - df['Year']

# Depreciation rate (tingkat depresiasi)
df['Depreciation_Rate'] = (df['Present_Price'] - df['Selling_Price']) / df['Present_Price']

# Price per km (harga per kilometer)
df['Price_per_km'] = df['Selling_Price'] / (df['Kms_Driven'] + 1)  # +1 untuk menghindari pembagian dengan 0

# Mileage category (kategori kilometer tempuh)
df['Mileage_Category'] = pd.cut(df['Kms_Driven'],
                               bins=[0, 20000, 50000, 100000, float('inf')],
                               labels=['Low', 'Medium', 'High', 'Very High'])

print("New features created successfully!")

"""Untuk meningkatkan performa model, kita membuat beberapa fitur baru:
- 'Car_Age': umur mobil berdasarkan tahun pembuatan
- 'Depreciation_Rate': tingkat depresiasi dari harga showroom
- 'Price_per_km': harga per kilometer tempuh
- 'Mileage_Category': kategori berdasarkan kilometer tempuh

### 4.3 Handling Outliers
"""

# Remove price outliers using IQR method
Q1 = df['Selling_Price'].quantile(0.25)
Q3 = df['Selling_Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Original dataset size: {len(df)}")
df_clean = df[(df['Selling_Price'] >= lower_bound) & (df['Selling_Price'] <= upper_bound)]
print(f"Dataset size after outlier removal: {len(df_clean)}")

"""- Untuk menangani outlier pada harga mobil, kita menggunakan metode IQR (Interquartile Range).
- Data yang berada di luar range (Q1 - 1.5*IQR) hingga (Q3 + 1.5*IQR) dianggap sebagai outlier dan dihapus.

### 4.4 Encoding Categorical Variables
"""

print("Encoding categorical variables...")

# Label encoding untuk variabel ordinal
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_transmission = LabelEncoder()
le_mileage = LabelEncoder()

df_clean['Fuel_Type_Encoded'] = le_fuel.fit_transform(df_clean['Fuel_Type'])
df_clean['Seller_Type_Encoded'] = le_seller.fit_transform(df_clean['Seller_Type'])
df_clean['Transmission_Encoded'] = le_transmission.fit_transform(df_clean['Transmission'])
df_clean['Mileage_Category_Encoded'] = le_mileage.fit_transform(df_clean['Mileage_Category'])

# One-hot encoding untuk Car_Name (brand)
df_clean['Car_Brand'] = df_clean['Car_Name'].str.split().str[0]
df_encoded = pd.get_dummies(df_clean, columns=['Car_Brand'], prefix='Brand')

print("Categorical encoding completed!")

"""### 4.5 Feature Selection"""

# Pilih fitur yang akan digunakan untuk modeling
feature_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner', 'Car_Age',
                  'Depreciation_Rate', 'Price_per_km', 'Fuel_Type_Encoded',
                  'Seller_Type_Encoded', 'Transmission_Encoded', 'Mileage_Category_Encoded']

# Tambahkan kolom brand yang sudah di-encode
brand_columns = [col for col in df_encoded.columns if col.startswith('Brand_')]
feature_columns.extend(brand_columns[:10])  # Ambil 10 brand teratas

# Buat correlation matrix untuk feature selection
corr_with_target = df_encoded[feature_columns + ['Selling_Price']].corr()['Selling_Price'].abs().sort_values(ascending=False)
print("\nTop features by correlation with Selling_Price:")
print(corr_with_target.head(10))

"""### 4.6 Feature Scaling"""

# Pilih fitur numerik untuk scaling
numeric_features = ['Year', 'Present_Price', 'Kms_Driven', 'Owner', 'Car_Age',
                   'Depreciation_Rate', 'Price_per_km']

scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

print("Feature scaling completed!")

"""- Normalisasi fitur numeric dilakukan menggunakan StandardScaler dari sklearn.
- Proses ini penting karena beberapa algoritma machine learning sensitif terhadap skala fitur.

### 4.7 Final Dataset Preparation
"""

# Prepare final features
final_features = feature_columns
X = df_encoded[final_features]
y = df_encoded['Selling_Price']

print(f"Final dataset shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

"""### 4.8 Split Data"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

"""- Data dibagi menjadi data training (80%) dan data testing (20%) dengan random_state=42
- untuk memastikan hasil yang konsisten dan dapat direproduksi.

## 5. Model Development

### 5.1 Model 1: Linear Regression
"""

print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

"""**Cara Kerja**:
- Linear Regression mencari hubungan linear antara fitur-fitur input dan target (harga mobil).
- Model ini meminimalkan sum of squared errors antara prediksi dan nilai aktual.

### 5.2 Model 2: Decision Tree Regressor
"""

print("Training Decision Tree model...")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

"""**Cara Kerja**:
- Decision Tree Regressor membagi data menjadi subset-subset yang lebih kecil
- berdasarkan fitur dan nilai threshold tertentu, membentuk struktur pohon.

### 5.3 Model 3: Random Forest Regressor
"""

print("Training Random Forest model...")

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomForest
rf = RandomForestRegressor(random_state=42)

# GridSearchCV
print("Performing hyperparameter tuning...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Fit model
grid_search.fit(X_train, y_train)

# Best model
best_rf_model = grid_search.best_estimator_

# Print best parameters
print("Best parameters for Random Forest:", grid_search.best_params_)

"""**Cara Kerja**:
- Random Forest Regressor adalah ensemble model yang terdiri dari banyak decision trees.
- Setiap tree dilatih pada subset data dan subset fitur yang berbeda.

## 6. Model Evaluation dan Comparison
"""

def evaluate_model(model, X_test, y_test, name='Model'):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"== {name} Evaluation ==")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print()

    return mae, rmse, r2

print("Evaluating all models...")
lr_mae, lr_rmse, lr_r2 = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
dt_mae, dt_rmse, dt_r2 = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
rf_mae, rf_rmse, rf_r2 = evaluate_model(best_rf_model, X_test, y_test, "Random Forest (Best)")

"""### Visualisasi Hasil Prediksi"""

plt.figure(figsize=(15, 5))

# Linear Regression
plt.subplot(1, 3, 1)
y_pred_lr = lr_model.predict(X_test)
plt.scatter(y_test, y_pred_lr, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression')
plt.grid(True)

# Decision Tree
plt.subplot(1, 3, 2)
y_pred_dt = dt_model.predict(X_test)
plt.scatter(y_test, y_pred_dt, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Decision Tree')
plt.grid(True)

# Random Forest
plt.subplot(1, 3, 3)
y_pred_rf = best_rf_model.predict(X_test)
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest (Best)')
plt.grid(True)

plt.tight_layout()
plt.show()

"""### Feature Importance Analysis"""

if hasattr(best_rf_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='Importance', y='Feature')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))

"""## 7. Model Testing dengan Contoh Data"""

# Contoh data mobil untuk testing
sample_data = {
    'Year': 2018,
    'Present_Price': 9.85,
    'Kms_Driven': 35000,
    'Fuel_Type': 'Petrol',
    'Seller_Type': 'Individual',
    'Transmission': 'Manual',
    'Owner': 0,
    'Selling_Price': 7.5  # Harga aktual untuk perbandingan
}

print("Testing model with sample data:")
print("Sample car specifications:")
for key, value in sample_data.items():
    if key != 'Selling_Price':
        print(f"  {key}: {value}")

# Simpan harga aktual
actual_price = sample_data['Selling_Price']

# Buat DataFrame dari sample data
sample_df = pd.DataFrame([sample_data])
sample_df = sample_df.drop(columns=['Selling_Price'])

# Feature engineering untuk sample data
sample_df['Car_Age'] = 2024 - sample_df['Year']
sample_df['Depreciation_Rate'] = (sample_df['Present_Price'] - actual_price) / sample_df['Present_Price']
sample_df['Price_per_km'] = actual_price / (sample_df['Kms_Driven'] + 1)
sample_df['Mileage_Category'] = pd.cut(sample_df['Kms_Driven'],
                                      bins=[0, 20000, 50000, 100000, float('inf')],
                                      labels=['Low', 'Medium', 'High', 'Very High'])

# Encoding
sample_df['Fuel_Type_Encoded'] = le_fuel.transform(sample_df['Fuel_Type'])
sample_df['Seller_Type_Encoded'] = le_seller.transform(sample_df['Seller_Type'])
sample_df['Transmission_Encoded'] = le_transmission.transform(sample_df['Transmission'])
sample_df['Mileage_Category_Encoded'] = le_mileage.transform(sample_df['Mileage_Category'])

# Pastikan semua kolom sesuai dengan training data
for col in X_train.columns:
    if col not in sample_df.columns:
        sample_df[col] = 0

# Pilih kolom sesuai urutan training
sample_df = sample_df[X_train.columns]

# Prediksi menggunakan model terbaik
predicted_price = best_rf_model.predict(sample_df)[0]

print(f"\nPrediction Results:")
print(f"Actual Price: ₹{actual_price:.2f} Lakh")
print(f"Predicted Price: ₹{predicted_price:.2f} Lakh")
print(f"Difference: ₹{abs(actual_price - predicted_price):.2f} Lakh")
print(f"Percentage Error: {abs(actual_price - predicted_price) / actual_price * 100:.2f}%")

"""## 8. Model Saving"""

import joblib

# Save the best model
joblib.dump(best_rf_model, 'car_price_prediction_model.pkl')
joblib.dump(scaler, 'car_price_scaler.pkl')
joblib.dump([le_fuel, le_seller, le_transmission, le_mileage], 'car_price_encoders.pkl')

print("Model and preprocessing objects saved successfully!")

"""## 9. Kesimpulan"""

print("\n" + "="*50)
print("PROJECT SUMMARY")
print("="*50)

print(f"\nDataset Information:")
print(f"- Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"- Final dataset size: {len(df_clean)} rows")
print(f"- Number of features used: {len(final_features)}")

print(f"\nModel Performance Comparison:")
print(f"Linear Regression    - MAE: {lr_mae:.4f}, RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}")
print(f"Decision Tree        - MAE: {dt_mae:.4f}, RMSE: {dt_rmse:.4f}, R²: {dt_r2:.4f}")
print(f"Random Forest (Best) - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")

print(f"\nBest Model: Random Forest with parameters:")
print(f"Parameters: {grid_search.best_params_}")

print(f"\nKey Insights:")
print(f"- Model dapat menjelaskan {rf_r2*100:.2f}% variasi dalam harga mobil bekas")
print(f"- Error rata-rata prediksi: ₹{rf_rmse:.4f} Lakh")
print(f"- Fitur terpenting: Present_Price, Car_Age, dan Kms_Driven")

if rf_rmse/y.mean() < 0.2:
    print(f"- ✅ Target tercapai: RMSE < 20% dari rata-rata harga")
else:
    print(f"- ❌ Target belum tercapai: RMSE > 20% dari rata-rata harga")