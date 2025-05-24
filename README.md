# **Laporan Proyek Machine Learning - Muhammad Razi Al Kindi Nadra**

## **Domain Proyek**

Pasar mobil bekas merupakan salah satu segmen otomotif yang mengalami pertumbuhan signifikan di Indonesia. Menurut data Gabungan Industri Kendaraan Bermotor Indonesia (GAIKINDO), transaksi mobil bekas mencapai jutaan unit setiap tahunnya. Namun, penentuan harga yang tepat untuk mobil bekas masih menjadi tantangan besar bagi penjual, pembeli, maupun dealer [1].

Ketidakpastian dalam menentukan harga mobil bekas dapat menyebabkan kerugian finansial yang signifikan. Penjual mungkin menetapkan harga terlalu rendah dan kehilangan keuntungan, sementara pembeli berisiko membayar terlalu mahal untuk kendaraan yang tidak sesuai dengan nilai pasarnya. Masalah ini diperparah oleh banyaknya faktor yang mempengaruhi harga mobil bekas, seperti tahun pembuatan, jarak tempuh, kondisi kendaraan, merek, dan tipe transmisi.

Teknologi machine learning telah terbukti efektif dalam menyelesaikan masalah prediksi harga pada berbagai domain. Penelitian yang dilakukan oleh Gegic et al. [2] menunjukkan bahwa algoritma ensemble seperti Random Forest dapat memberikan akurasi tinggi dalam memprediksi harga mobil bekas. Studi lain oleh Pudaruth [3] mengonfirmasi bahwa pendekatan machine learning dapat mengidentifikasi pola kompleks dalam data otomotif yang sulit dideteksi secara manual.

Penerapan model prediksi harga mobil bekas tidak hanya bermanfaat untuk individu, tetapi juga untuk industri otomotif secara keseluruhan. Menurut Sun et al. [4], model prediksi yang akurat dapat meningkatkan transparansi pasar, mengurangi asimetri informasi, dan membantu dealer dalam strategi penetapan harga yang optimal.

[1] [GAIKINDO - Gabungan Industri Kendaraan Bermotor Indonesia](https://www.gaikindo.or.id/)  
[2] [Gegic, E., et al. (2019). Car Price Prediction using Machine Learning Techniques](https://www.researchgate.net/publication/334793013_Car_Price_Prediction_using_Machine_Learning_Techniques)  
[3] [Pudaruth, S. (2014). Predicting the Price of Used Cars using Machine Learning Techniques](https://www.researchgate.net/publication/265073974_Predicting_the_Price_of_Used_Cars_using_Machine_Learning_Techniques)  
[4] [Sun, N., et al. (2017). Applications of Machine Learning Algorithms for Used Car Price Prediction](https://scholar.google.com/scholar?q=Applications+of+Machine+Learning+Algorithms+for+Used+Car+Price+Prediction)

## **Business Understanding**

### **Problem Statements**

Beberapa permasalahan yang akan diselesaikan dalam proyek ini:

- Bagaimana mengembangkan model machine learning yang dapat memprediksi harga mobil bekas secara akurat berdasarkan spesifikasi dan kondisi kendaraan?
- Faktor-faktor apa saja yang paling berpengaruh terhadap penentuan harga mobil bekas di pasar?
- Seberapa akurat model machine learning dapat memprediksi harga mobil bekas dibandingkan dengan harga aktual di pasar?

### **Goals**

Tujuan dari proyek ini adalah:

- Mengembangkan model prediksi harga mobil bekas yang memiliki tingkat error rendah menggunakan algoritma machine learning.
- Mengidentifikasi fitur-fitur yang memiliki korelasi dan pengaruh paling signifikan terhadap harga mobil bekas.
- Menghasilkan model prediksi yang memiliki error (RMSE) kurang dari 20% dari rata-rata harga mobil bekas dalam dataset.

### **Solution Statements**

Untuk mencapai tujuan yang telah ditetapkan, beberapa solusi yang akan diimplementasikan:

1. Menggunakan algoritma Linear Regression sebagai baseline model untuk memprediksi harga mobil bekas karena algoritma ini sederhana dan mudah diinterpretasi untuk memahami hubungan linear antar variabel.

2. Mengimplementasikan algoritma Decision Tree Regressor yang mampu menangkap hubungan non-linear antara fitur kendaraan dan harga jual.

3. Menerapkan algoritma Random Forest Regressor yang dapat menangani berbagai tipe data dan mampu menangkap interaksi kompleks antar fitur dengan tingkat akurasi yang tinggi.

4. Melakukan hyperparameter tuning pada model terbaik menggunakan GridSearchCV untuk mengoptimalkan performa prediksi.

5. Menggunakan metrik evaluasi MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), dan R-squared untuk mengukur dan membandingkan performa model.

## **Data Understanding**

Dataset yang digunakan dalam proyek ini adalah Car Price Prediction dataset yang tersedia di [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho). Dataset ini berisi informasi penjualan mobil bekas dari platform CardekHo, salah satu marketplace mobil bekas terbesar di India. Dataset terdiri dari 301 records dan 9 columns (fitur).

### **Variabel-variabel pada Car Price Prediction dataset adalah sebagai berikut:**

1. **Car_Name**: Nama dan model mobil (string)
2. **Year**: Tahun pembuatan mobil (integer)
3. **Selling_Price**: Harga jual mobil bekas dalam satuan Lakh Rupee India (float) - **Target Variable**
4. **Present_Price**: Harga mobil baru saat ini di showroom dalam satuan Lakh Rupee India (float)
5. **Kms_Driven**: Jarak tempuh kendaraan dalam kilometer (integer)
6. **Fuel_Type**: Jenis bahan bakar (Petrol/Diesel/CNG) (string)
7. **Seller_Type**: Jenis penjual (Dealer/Individual) (string)
8. **Transmission**: Jenis transmisi (Manual/Automatic) (string)
9. **Owner**: Jumlah pemilik sebelumnya (integer)

### **Exploratory Data Analysis**

Untuk memahami dataset dengan lebih baik, dilakukan analisis eksplorasi terhadap data. Berikut beberapa insight yang diperoleh:

#### **Struktur Data**

Dataset memiliki 301 baris dan 9 kolom. Dari hasil pengecekan tipe data, terdapat campuran antara data numerik (Year, Selling_Price, Present_Price, Kms_Driven, Owner) dan data kategorikal (Car_Name, Fuel_Type, Seller_Type, Transmission).

```javascript
Shape: (301, 9)

Data Types:
Car_Name         object
Year              int64
Selling_Price   float64
Present_Price   float64
Kms_Driven        int64
Fuel_Type        object
Seller_Type      object
Transmission     object
Owner             int64
dtype: object
```

#### **Statistik Deskriptif**

Statistik deskriptif menunjukkan bahwa rata-rata harga jual mobil bekas adalah sekitar **4.66 Lakh Rupee**, dengan harga minimum **0.1 Lakh** dan maksimum **35 Lakh Rupee**. Rata-rata tahun pembuatan mobil adalah sekitar **2013**, dengan jarak tempuh rata-rata sekitar **36,947 kilometer**. Sebagian besar mobil (75%) memiliki harga di bawah 6 Lakh Rupee.

```javascript
Statistical Summary:
              Year  Selling_Price  Present_Price     Kms_Driven       Owner
count   301.000000     301.000000     301.000000     301.000000  301.000000
mean   2013.627907       4.661296       7.628472   36947.205980    0.043189
std       2.891554       5.082812       8.644115   38886.883882    0.247915
min    2003.000000       0.100000       0.320000     500.000000    0.000000
25%    2012.000000       0.900000       1.200000   15000.000000    0.000000
50%    2014.000000       3.600000       6.400000   32000.000000    0.000000
75%    2016.000000       6.000000       9.900000   48767.000000    0.000000
max    2018.000000      35.000000      92.600000  500000.000000    3.000000
```

#### **Kondisi Data**

- **Missing Values**: Dataset tidak memiliki missing values pada semua kolom
- **Duplicate Values**: Terdapat **2 baris data duplikat** yang perlu ditangani dalam tahap preprocessing
- **Data Quality**: Semua kolom memiliki tipe data yang sesuai dan tidak ada anomali format

#### **Distribusi Harga Mobil**

Analisis distribusi harga mobil bekas menunjukkan bahwa sebagian besar mobil memiliki harga antara 2-8 Lakh Rupee, dengan beberapa mobil mewah yang memiliki harga di atas 20 Lakh Rupee. Distribusi harga cenderung right-skewed (condong ke kanan).

#### **Korelasi antar Fitur**

Analisis korelasi menunjukkan bahwa fitur Present_Price (harga showroom saat ini) memiliki korelasi positif yang sangat kuat dengan Selling_Price. Fitur Year juga menunjukkan korelasi positif yang signifikan, sementara Kms_Driven memiliki korelasi negatif dengan harga jual.

#### **Analisis Fitur Kategorikal**

- **Fuel_Type**: Mayoritas mobil menggunakan bahan bakar Petrol, diikuti Diesel, dan sedikit yang menggunakan CNG
- **Seller_Type**: Sebagian besar penjualan dilakukan oleh Dealer dibandingkan Individual
- **Transmission**: Mobil dengan transmisi Manual lebih dominan dibandingkan Automatic

#### **Pengaruh Tahun Pembuatan**

Mobil yang lebih baru (tahun pembuatan lebih tinggi) cenderung memiliki harga jual yang lebih tinggi, menunjukkan adanya depresiasi nilai kendaraan seiring waktu.

#### **Pengaruh Jarak Tempuh**

Terdapat korelasi negatif antara jarak tempuh (Kms_Driven) dengan harga jual, di mana mobil dengan kilometer yang lebih rendah cenderung memiliki harga yang lebih tinggi.

## **Data Preparation**

Tahap data preparation dilakukan secara sistematis untuk mempersiapkan data sebelum pemodelan:

### **1. Penanganan Missing Values**

Pengecekan missing values menunjukkan bahwa dataset tidak memiliki nilai yang hilang pada semua kolom. Namun, dilakukan validasi ulang untuk memastikan kualitas data dan konsistensi format pada setiap kolom.

```javascript
Missing Values Check:
Car_Name         0
Year             0
Selling_Price    0
Present_Price    0
Kms_Driven       0
Fuel_Type        0
Seller_Type      0
Transmission     0
Owner            0
```

### **2. Feature Engineering**

Untuk meningkatkan kemampuan prediktif model, beberapa fitur baru dibuat berdasarkan domain knowledge otomotif:

- **Car_Age**: Menghitung usia mobil berdasarkan selisih antara tahun saat ini (2024) dengan tahun pembuatan mobil. Fitur ini lebih intuitif dibandingkan menggunakan tahun pembuatan langsung.

- **Depreciation_Rate**: Menghitung tingkat depresiasi dengan rumus (Present_Price - Selling_Price) / Present_Price. Fitur ini menunjukkan seberapa besar nilai mobil telah menyusut dari harga barunya.

- **Price_per_km**: Menghitung harga per kilometer tempuh dengan membagi Selling_Price dengan Kms_Driven (+1 untuk menghindari pembagian dengan nol). Fitur ini memberikan perspektif efisiensi nilai kendaraan.

- **Mileage_Category**: Mengkategorikan jarak tempuh menjadi Low (0-20,000 km), Medium (20,001-50,000 km), High (50,001-100,000 km), dan Very High (>100,000 km).

### **3. Penanganan Outliers**

Outliers pada variabel target (Selling_Price) dapat mempengaruhi performa model. Metode IQR (Interquartile Range) digunakan untuk mendeteksi dan menghapus outliers:

1. Menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3) dari distribusi harga jual
2. Menghitung IQR sebagai selisih antara Q3 dan Q1
3. Menentukan batas bawah (Q1 - 1.5*IQR) dan batas atas (Q3 + 1.5*IQR)
4. Menghapus data yang berada di luar rentang tersebut

Proses ini menghapus **17 data** yang dianggap sebagai outliers (dari 301 menjadi 284 data), menyisakan dataset yang lebih representatif untuk pemodelan.

```javascript
Original dataset size: 301
Dataset size after outlier removal: 284
```

### **4. Encoding Categorical Variables**

Transformasi variabel kategorikal menjadi format numerik dilakukan menggunakan dua pendekatan:

- **Label Encoding**: Digunakan untuk variabel Fuel_Type, Seller_Type, Transmission, dan Mileage_Category. Metode ini mengubah kategori menjadi angka integer.

- **One-Hot Encoding**: Digunakan untuk Car_Brand (yang diekstrak dari Car_Name). Metode ini membuat kolom biner untuk setiap merek mobil, memungkinkan model memahami pengaruh merek terhadap harga.

### **5. Feature Selection**

Pemilihan fitur dilakukan berdasarkan korelasi dengan target. Dari hasil analisis korelasi, fitur-fitur dengan korelasi tertinggi dengan Selling_Price adalah:

```javascript
Top features by correlation with Selling_Price:
Selling_Price          1.000000
Present_Price          0.801950
Seller_Type_Encoded    0.750623
Fuel_Type_Encoded      0.421596
Price_per_km           0.351048
Brand_Bajaj            0.324781
Depreciation_Rate      0.289842
Car_Age                0.279578
Year                   0.279578
Brand_Honda            0.271217
```

Fitur-fitur ini menunjukkan bahwa Present_Price, Seller_Type, dan Fuel_Type memiliki pengaruh yang signifikan terhadap harga jual mobil bekas.

### **6. Feature Scaling**

Normalisasi fitur numerik dilakukan menggunakan StandardScaler untuk memastikan semua fitur memiliki kontribusi seimbang dalam model. Fitur-fitur seperti Present_Price, Kms_Driven, dan Car_Age memiliki skala yang berbeda, sehingga standardisasi diperlukan untuk algoritma yang sensitif terhadap skala seperti Linear Regression.

### **7. Final Dataset Preparation**

Dataset final disusun dengan menggabungkan seluruh fitur yang telah diproses. Dataset final memiliki dimensi **(284, 21)** dengan 21 fitur yang siap untuk pemodelan.

```javascript
Final dataset shape: (284, 21)
Target variable shape: (284,)
```

### **8. Split Data**

Dataset dibagi menjadi data training (80%) dan data testing (20%) menggunakan train_test_split dengan random_state=42 untuk memastikan reproducibility.

```javascript
Training set size: (227, 21)
Test set size: (57, 21)
```

Pembagian ini memungkinkan evaluasi objektif terhadap kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya.

## **Modeling**

Pada tahap ini, tiga algoritma machine learning diterapkan untuk memprediksi harga mobil bekas: Linear Regression, Decision Tree Regressor, dan Random Forest Regressor.

### **1. Linear Regression**

Model Linear Regression diimplementasikan sebagai baseline model karena kesederhanaan dan kemudahan interpretasinya.

**Cara Kerja**: Linear Regression mencari hubungan linear terbaik antara fitur-fitur input dan target (harga mobil bekas) dengan meminimalkan sum of squared errors. Model menghasilkan persamaan linear dengan koefisien yang menunjukkan kontribusi setiap fitur terhadap harga.

**Parameter**: Menggunakan parameter default dari scikit-learn tanpa modifikasi khusus.

**Kelebihan Linear Regression:**

- Mudah diimplementasikan dan diinterpretasi
- Komputasi yang efisien dan cepat
- Memberikan insight langsung tentang pengaruh setiap fitur
- Tidak mudah overfitting pada dataset kecil

**Kekurangan Linear Regression:**

- Hanya dapat menangkap hubungan linear
- Sensitif terhadap outliers
- Asumsi linearitas mungkin tidak sesuai untuk data harga mobil yang kompleks
- Tidak dapat menangkap interaksi antar fitur

### **2. Decision Tree Regressor**

Model Decision Tree Regressor dapat menangkap hubungan non-linear dalam data harga mobil bekas.

**Cara Kerja**: Decision Tree membagi data menjadi subset-subset berdasarkan nilai threshold pada fitur tertentu, membentuk struktur pohon keputusan. Setiap leaf node berisi prediksi harga berdasarkan rata-rata nilai target dalam subset tersebut.

**Parameter yang digunakan:**

- random_state=42 untuk reproducibility
- Parameter lain menggunakan nilai default (criterion='squared_error', max_depth=None)

**Kelebihan Decision Tree Regressor:**

- Dapat menangkap hubungan non-linear dan interaksi antar fitur
- Mudah divisualisasikan dan diinterpretasi
- Tidak memerlukan feature scaling
- Dapat menangani data campuran (numerik dan kategorikal)

**Kekurangan Decision Tree Regressor:**

- Rentan terhadap overfitting, terutama pada dataset kecil
- Tidak stabil (perubahan kecil pada data dapat mengubah struktur pohon)
- Bias terhadap fitur dengan banyak nilai unik
- Performanya seringkali tidak sebaik model ensemble

### **3. Random Forest Regressor**

Model Random Forest Regressor adalah model ensemble yang menggabungkan prediksi dari multiple decision trees.

**Cara Kerja**: Random Forest melatih banyak decision trees pada subset data yang berbeda (bootstrap sampling) dan subset fitur yang berbeda (feature bagging). Prediksi final diperoleh dengan merata-ratakan hasil dari semua trees, yang mengurangi variance dan meningkatkan generalisasi.

**Hyperparameter Tuning**: Dilakukan optimasi parameter menggunakan GridSearchCV dengan parameter grid:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

**Parameter Terbaik yang Diperoleh:**

```javascript
Best parameters for Random Forest: 
{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
```

- **n_estimators**: 100 (jumlah trees dalam forest)
- **max_depth**: 10 (kedalaman maksimum setiap tree)
- **min_samples_split**: 2 (minimal sampel untuk split internal node)
- **min_samples_leaf**: 1 (minimal sampel di setiap leaf node)

**Kelebihan Random Forest Regressor:**

- Performa yang superior dibandingkan single decision tree
- Mengurangi risiko overfitting melalui ensemble averaging
- Dapat menangani fitur dengan berbagai skala dan tipe
- Memberikan feature importance yang berguna untuk interpretasi
- Robust terhadap outliers dan noise

**Kekurangan Random Forest Regressor:**

- Lebih kompleks dan membutuhkan lebih banyak computational resources
- Lebih sulit diinterpretasi dibandingkan single decision tree
- Training time yang lebih lama
- Dapat mengalami overfitting jika parameter tidak diatur dengan baik

### **4. Model Selection dan Improvement**

Berdasarkan hasil evaluasi, Random Forest Regressor dengan hyperparameter tuning dipilih sebagai model terbaik karena memberikan performa superior dalam hal akurasi prediksi. Proses hyperparameter tuning menggunakan 5-fold cross validation memastikan bahwa parameter yang dipilih dapat memberikan performa yang konsisten pada data yang berbeda.

## **Evaluation**

Evaluasi model dilakukan menggunakan beberapa metrik yang relevan untuk masalah regresi:

### **1. Mean Absolute Error (MAE)**

MAE mengukur rata-rata dari nilai absolut selisih antara nilai aktual dan prediksi.

**Formula:**

```javascript
MAE = (1/n) * Σ|yi - ŷi|
```

**Interpretasi**: MAE memberikan gambaran rata-rata kesalahan prediksi dalam unit yang sama dengan target (Lakh Rupee). Nilai yang lebih rendah menunjukkan performa yang lebih baik.

### **2. Mean Squared Error (MSE)**

MSE mengukur rata-rata dari kuadrat selisih antara nilai aktual dan prediksi.

**Formula:**

```javascript
MSE = (1/n) * Σ(yi - ŷi)²
```

**Interpretasi**: MSE memberikan penalti yang lebih besar untuk error yang besar karena proses pengkuadratan. Metrik ini sensitif terhadap outliers.

### **3. Root Mean Squared Error (RMSE)**

RMSE adalah akar kuadrat dari MSE, memberikan error dalam unit yang sama dengan target.

**Formula:**

```javascript
RMSE = √MSE = √[(1/n) * Σ(yi - ŷi)²]
```

**Interpretasi**: RMSE lebih mudah diinterpretasi dibandingkan MSE karena memiliki unit yang sama dengan variabel target. Nilai yang lebih rendah menunjukkan prediksi yang lebih akurat.

### **4. R-squared (R²)**

R² mengukur proporsi variasi dalam variabel target yang dapat dijelaskan oleh model.

**Formula:**

```javascript
R² = 1 - (SSres/SStot)
dimana:
SSres = Σ(yi - ŷi)²
SStot = Σ(yi - ȳ)²
```

**Interpretasi**: R² bernilai antara 0 dan 1. Nilai mendekati 1 menunjukkan model yang sangat baik dalam menjelaskan variasi data.

### **Hasil Evaluasi Model**

Berikut adalah perbandingan performa ketiga model yang diimplementasikan:

```javascript
== Linear Regression Evaluation ==
MAE:  0.7621
RMSE: 1.2360
R²:   0.8373

== Decision Tree Evaluation ==
MAE:  0.4177
RMSE: 0.6001
R²:   0.9617

== Random Forest (Best) Evaluation ==
MAE:  0.3255
RMSE: 0.5135
R²:   0.9719
```

 Model  MAE  RMSE  R²  Linear Regression  0.7621  1.2360  0.8373  Decision Tree  0.4177  0.6001  0.9617  Random Forest (Tuned)  0.3255  0.5135  0.9719 

**Analisis Hasil:**

1. **Random Forest (Tuned)** memberikan performa terbaik dengan:

- MAE terendah (0.3255 Lakh Rupee)
- RMSE terendah (0.5135 Lakh Rupee)  
- R² tertinggi (0.9719)

2. **Decision Tree** menunjukkan performa yang sangat baik:

- R² mencapai 0.9617, menjelaskan 96.17% variasi data
- Signifikan lebih baik dari Linear Regression

3. **Linear Regression** sebagai baseline model memberikan performa yang cukup baik:

- R² 0.8373 menunjukkan model dapat menjelaskan 83.73% variasi
- Performa yang lebih rendah mengindikasikan adanya hubungan non-linear dalam data

### **Feature Importance Analysis**

Analisis feature importance dari Random Forest menunjukkan kontribusi setiap fitur:

```javascript
Top 10 Most Important Features:
                     Feature  Importance
1              Present_Price    0.765271
6               Price_per_km    0.166787
5          Depreciation_Rate    0.038752
8        Seller_Type_Encoded    0.009389
0                       Year    0.007965
4                    Car_Age    0.006328
2                 Kms_Driven    0.003867
7          Fuel_Type_Encoded    0.000673
10  Mileage_Category_Encoded    0.000490
9       Transmission_Encoded    0.000296
```

**Top 5 Most Important Features:**

1. **Present_Price** (76.5%) - Harga showroom saat ini
2. **Price_per_km** (16.7%) - Harga per kilometer tempuh
3. **Depreciation_Rate** (3.9%) - Tingkat depresiasi
4. **Seller_Type_Encoded** (0.9%) - Jenis penjual
5. **Year** (0.8%) - Tahun pembuatan

Hasil ini mengonfirmasi bahwa **harga showroom saat ini** adalah faktor dominan yang mempengaruhi harga mobil bekas, diikuti oleh **efisiensi nilai per kilometer** dan **tingkat depresiasi**.

### **Model Testing dengan Data Sampel**

Pengujian model dilakukan pada sampel mobil dengan spesifikasi:

```javascript
Sample car specifications:
  Year: 2018
  Present_Price: 9.85
  Kms_Driven: 35000
  Fuel_Type: Petrol
  Seller_Type: Individual
  Transmission: Manual
  Owner: 0
```

**Hasil Prediksi:**

```javascript
Prediction Results:
Actual Price: ₹7.50 Lakh
Predicted Price: ₹8.55 Lakh
Difference: ₹1.05 Lakh
Percentage Error: 14.04%
```

Error sebesar 14.04% masih dalam batas yang dapat diterima untuk prediksi harga mobil bekas, menunjukkan model memiliki akurasi yang baik dalam memprediksi harga pada data yang belum pernah dilihat sebelumnya.

### **Evaluasi Pencapaian Goals**

Berdasarkan hasil evaluasi:

1. ✅ **Goal 1 Tercapai**: Model Random Forest berhasil dikembangkan dengan tingkat error yang rendah (RMSE = 0.5135 Lakh Rupee)

2. ✅ **Goal 2 Tercapai**: Fitur-fitur penting telah diidentifikasi, dengan Present_Price, Price_per_km, dan Depreciation_Rate sebagai prediktor utama

3. ✅ **Goal 3 Tercapai**: RMSE model (0.5135) kurang dari 20% dari rata-rata harga mobil dalam dataset (4.66 Lakh Rupee), yaitu sekitar **11.02%**

## **Kesimpulan**

Proyek prediksi harga mobil bekas ini telah berhasil mengembangkan model machine learning yang akurat dan dapat diandalkan. Berikut adalah kesimpulan utama:

### **1. Performa Model**

- **Model Random Forest dengan hyperparameter tuning** memberikan performa terbaik dengan **R² = 0.9719** dan **RMSE = 0.5135 Lakh Rupee**
- Model mampu menjelaskan **97.19% variasi** dalam harga mobil bekas dengan error rata-rata hanya sekitar ₹0.51 Lakh
- Pengujian pada data sampel menunjukkan error prediksi **14.04%**, yang masih dalam batas toleransi yang baik untuk aplikasi praktis

### **2. Faktor Penentu Harga**

Analisis feature importance mengidentifikasi faktor-faktor utama yang mempengaruhi harga mobil bekas:

- **Present Price (76.5%)**: Harga showroom saat ini menjadi prediktor dominan yang sangat mempengaruhi harga jual
- **Price per km (16.7%)**: Efisiensi nilai kendaraan per kilometer tempuh menunjukkan tingkat pemakaian
- **Depreciation Rate (3.9%)**: Tingkat penurunan nilai dari harga baru memberikan gambaran depresiasi
- **Seller Type (0.9%)**: Jenis penjual (dealer vs individual) mempengaruhi kepercayaan dan harga
- **Year (0.8%)**: Tahun pembuatan kendaraan mencerminkan teknologi dan kondisi

### **3. Insights Bisnis**

- **Dominasi Harga Showroom**: Present_Price memiliki pengaruh yang sangat dominan (76.5%), menunjukkan bahwa harga mobil bekas sangat bergantung pada harga referensi showroom
- **Efisiensi Pemakaian**: Price_per_km menjadi faktor kedua terpenting, mengindikasikan bahwa pembeli mempertimbangkan efisiensi nilai terhadap pemakaian kendaraan
- **Pengaruh Depresiasi**: Tingkat depresiasi yang terukur memberikan insight tentang seberapa cepat kendaraan kehilangan nilainya
- **Faktor Sekunder**: Meskipun penting, faktor seperti jenis bahan bakar, transmisi, dan kategori jarak tempuh memiliki pengaruh yang relatif kecil

### **4. Keunggulan Model**

- **Akurasi Tinggi**: Model mencapai akurasi prediksi **97.19%** dengan error rata-rata hanya **11.02%** dari rata-rata harga
- **Robustness**: Random Forest terbukti robust terhadap outliers dan dapat menangani data dengan berbagai skala
- **Interpretabilitas**: Feature importance memberikan insight yang jelas dan actionable tentang faktor-faktor yang mempengaruhi harga
- **Generalisasi**: Model menunjukkan performa konsisten pada data testing, mengindikasikan kemampuan generalisasi yang baik

### **5. Aplikasi Praktis**

Model ini dapat dimanfaatkan oleh berbagai stakeholder:

- **Penjual Individual**: Menentukan harga jual yang kompetitif dan realistis berdasarkan kondisi kendaraan
- **Dealer Mobil Bekas**: Strategi pricing yang optimal dan manajemen inventory yang lebih efektif
- **Pembeli**: Evaluasi kewajaran harga sebelum melakukan pembelian untuk menghindari overpaying
- **Lembaga Keuangan**: Penilaian agunan yang lebih akurat untuk kredit kendaraan bermotor
- **Platform Online**: Implementasi sistem rekomendasi harga otomatis untuk marketplace mobil bekas

### **6. Keterbatasan dan Pengembangan Masa Depan**

**Keterbatasan:**

- Dataset terbatas pada 284 sampel setelah cleaning, perlu dataset yang lebih besar untuk generalisasi yang lebih baik
- Tidak mempertimbangkan faktor eksternal seperti kondisi ekonomi, inflasi, atau tren pasar otomotif
- Fitur kondisi fisik kendaraan belum detail (kondisi mesin, body, interior, riwayat kecelakaan)
- Data terbatas pada pasar India, mungkin tidak langsung applicable untuk pasar Indonesia

**Saran Pengembangan:**

- **Ekspansi Dataset**: Menggunakan dataset yang lebih besar dan beragam untuk meningkatkan robustness dan generalisasi model
- **Feature Engineering Lanjutan**: Menambahkan fitur seperti riwayat servis, kondisi fisik detail, popularitas model, dan seasonal trends
- **Model Ensemble**: Mengeksplorasi kombinasi algoritma lain seperti XGBoost, LightGBM, atau CatBoost untuk performa yang lebih baik
- **Real-time Updates**: Implementasi sistem yang dapat mengupdate model berdasarkan tren pasar dan data penjualan terkini
- **Regional Adaptation**: Menyesuaikan model untuk karakteristik pasar lokal Indonesia dengan mempertimbangkan preferensi konsumen dan kondisi jalan

### **7. Dampak dan Manfaat**

Implementasi model prediksi harga mobil bekas ini diharapkan dapat:

- **Meningkatkan Transparansi Pasar**: Mengurangi asimetri informasi antara penjual dan pembeli dengan memberikan referensi harga yang objektif
- **Efisiensi Transaksi**: Mempercepat proses negosiasi dengan referensi harga yang dapat dipercaya dan berbasis data
- **Pengambilan Keputusan**: Membantu stakeholder membuat keputusan investasi dan pembelian yang lebih informed dan rasional
- **Standardisasi Industri**: Memberikan benchmark yang konsisten untuk penilaian harga mobil bekas di industri otomotif
- **Digitalisasi Sektor**: Mendorong transformasi digital dalam industri otomotif, khususnya segmen kendaraan bekas

Model prediksi harga mobil bekas yang telah dikembangkan menunjukkan potensi besar dalam transformasi digital industri otomotif. Dengan akurasi tinggi (97.19%) dan interpretabilitas yang baik, model ini siap untuk diimplementasikan dalam aplikasi praktis yang dapat memberikan value signifikan bagi seluruh ekosistem otomotif, dari individual hingga institusi besar.
