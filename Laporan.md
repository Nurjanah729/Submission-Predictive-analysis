# Laporan Proyek Machine Learning - Siti Nurjanah

## 1. Domain Proyek

### 1.1. Latar Belakang

Pendidikan tinggi merupakan salah satu pilar utama dalam pembangunan sumber daya manusia suatu negara. Seleksi masuk perguruan tinggi, seperti Ujian Tulis Berbasis Komputer (UTBK) di Indonesia, menjadi krusial dalam menentukan kualitas mahasiswa yang akan menempuh pendidikan di jenjang selanjutnya. Proses seleksi ini melibatkan ribuan bahkan ratusan ribu peserta setiap tahunnya, menghasilkan sejumlah besar data berupa nilai-nilai ujian. Pemanfaatan data ini secara cerdas dapat memberikan wawasan yang mendalam tentang faktor-faktor penentu kelulusan.

Proyek ini berangkat dari kebutuhan untuk memahami lebih dalam bagaimana nilai-nilai mata pelajaran UTBK Soshum 2020 memengaruhi status kelulusan peserta. Dengan menganalisis data historis, kita dapat membangun sistem prediksi yang tidak hanya mempermudah proses evaluasi, tetapi juga berpotensi memberikan umpan balik dini bagi calon mahasiswa atau bahkan bagi institusi pendidikan untuk merancang program dukungan yang lebih efektif.

### 1.2. Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

Masalah prediksi kelulusan adalah masalah klasifikasi biner yang relevan dalam konteks pendidikan. Saat ini, penentuan kelulusan biasanya didasarkan pada ambang batas skor yang ditetapkan. Namun, dengan pendekatan *machine learning*, kita dapat:
1.  **Mengidentifikasi Prediktor Utama:** Menemukan kombinasi nilai mata pelajaran atau kemampuan yang paling berkorelasi kuat dengan kelulusan, yang mungkin tidak selalu intuitif.
2.  **Optimasi Proses Seleksi:** Membangun model yang dapat memprediksi kelulusan dengan tingkat akurasi tinggi, mengurangi beban kerja manual, dan mempercepat proses penentuan status.
3.  **Intervensi Dini:** Memberikan informasi bagi peserta yang berpotensi "Tidak Lolos" agar dapat mengambil tindakan korektif atau mencari jalur lain lebih awal.
4.  **Alokasi Sumber Daya:** Institusi dapat mengalokasikan sumber daya (misalnya, beasiswa, program bimbingan) secara lebih efisien berdasarkan prediksi probabilitas kelulusan.

Masalah ini akan diselesaikan dengan membangun model klasifikasi Machine Learning yang dilatih pada dataset historis nilai UTBK dan status kelulusan. Model akan belajar pola dari data pelatihan untuk kemudian memprediksi status kelulusan pada data baru.


## 2. Business Understanding

### 2.1. Problem Statements (Pernyataan Masalah)

1.  Bagaimana kita dapat membangun model Machine Learning yang mampu memprediksi status kelulusan peserta UTBK Soshum 2020 (Lolos atau Tidak Lolos) berdasarkan nilai-nilai dari berbagai mata pelajaran dan kemampuan?
2.  Fitur (nilai mata pelajaran/kemampuan) apa yang paling berkorelasi dan memiliki dampak signifikan terhadap status kelulusan peserta UTBK Soshum 2020?
3.  Di antara model klasifikasi yang diuji, model mana yang memberikan performa terbaik dalam memprediksi status kelulusan ini, terutama mengingat adanya ketidakseimbangan kelas pada data target?

### 2.2. Goals (Tujuan Proyek)

1.  Mengembangkan model Machine Learning klasifikasi dengan akurasi tinggi (target `accuracy_score` dan `f1_score` di atas 0.90) untuk memprediksi status kelulusan peserta UTBK Soshum 2020.
2.  Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap kelulusan melalui analisis *feature importance*.
3.  Memilih model terbaik berdasarkan metrik evaluasi yang relevan (khususnya *F1-Score* untuk menangani *class imbalance*) sebagai solusi akhir.

### 2.3. Solution Statements

Untuk mencapai tujuan di atas, beberapa pendekatan solusi akan diterapkan:

1.  **Penggunaan Multiple Algoritma Klasifikasi:** Proyek ini akan mengimplementasikan dan membandingkan kinerja dari setidaknya dua algoritma klasifikasi yang berbeda dan satu model *baseline*. Algoritma yang dipilih adalah **Logistic Regression** (sebagai model *baseline*), **Random Forest Classifier**, dan **XGBoost Classifier**. Pendekatan ini memungkinkan perbandingan performa antar model untuk mengidentifikasi yang paling cocok dengan karakteristik dataset.
2.  **Optimasi Model Melalui Hyperparameter Tuning:** Untuk Random Forest dan XGBoost, proses *hyperparameter tuning* (misalnya menggunakan `GridSearchCV` atau `RandomizedSearchCV`) akan dilakukan. Hal ini bertujuan untuk mengoptimalkan parameter internal model agar mencapai kinerja prediksi terbaik pada data yang tersedia, meningkatkan performa di atas model dengan parameter *default* atau *baseline* yang kurang optimal.
3.  **Pengukuran Kinerja Model dengan Metrik Evaluasi Terukur:** Kinerja setiap model akan diukur menggunakan metrik evaluasi yang relevan untuk masalah klasifikasi biner dengan *class imbalance*, yaitu:
    * **Accuracy Score:** Mengukur proporsi prediksi yang benar dari total prediksi.
    * **Precision:** Mengukur seberapa akurat model saat memprediksi kelas positif.
    * **Recall:** Mengukur kemampuan model untuk menemukan semua sampel positif.
    * **F1-Score:** Rata-rata harmonik dari Precision dan Recall, sangat berguna untuk *class imbalance*.
    * **Confusion Matrix:** Memberikan gambaran visual tentang True Positives, True Negatives, False Positives, dan False Negatives.
    Pemilihan F1-Score dan analisis Confusion Matrix sangat penting karena dataset memiliki ketidakseimbangan kelas, di mana akurasi saja bisa misleading.

---

## 3. Data Understanding

### 3.1. Sumber Data

**Tautan Sumber Data:** `https://www.kaggle.com/datasets/yanisuprayitno/datase-skor-utbk-soshum-2020`

### 3.2. Informasi Data

* **Jumlah Sampel Awal:** Dataset memiliki **36.201 baris** (sampel peserta).
* **Jumlah Fitur Awal:** Dataset memiliki **13 kolom** (fitur).
* **Kondisi Data:**
    * Tidak ada *missing values* yang terdeteksi.
    * Tidak ada baris duplikat yang ditemukan.
    * Terdapat **ketidakseimbangan kelas** pada variabel target 'Keterangan':
        * **Lolos:** Sekitar **83.33%** dari total peserta.
        * **Tidak Lolos:** Sekitar **16.67%** dari total peserta.

### 3.3. Variabel/Fitur pada Data

| Fitur                             | Deskripsi                                                          | Tipe Data Awal | Tipe Data Setelah Cleaning/Encoding |
| :-------------------------------- | :----------------------------------------------------------------- | :------------- | :---------------------------------- |
| `absen`                           | Nomor absen peserta (akan dihapus)                                 | `int64`        | (Dihapus)                           |
| `NISN`                            | Nomor Induk Siswa Nasional (akan dihapus)                          | `int64`        | (Dihapus)                           |
| `NPSN`                            | Nomor Pokok Sekolah Nasional (akan dihapus)                        | `int64`        | (Dihapus)                           |
| `Ekonomi`                         | Nilai UTBK mata pelajaran Ekonomi                                  | `int64`        | `float64` (setelah scaling)         |
| `Geografi`                        | Nilai UTBK mata pelajaran Geografi                                 | `int64`        | `float64` (setelah scaling)         |
| `Kemampuan Bacaan dan Menulis`    | Nilai UTBK Kemampuan Bacaan dan Menulis                            | `int64`        | `float64` (setelah scaling)         |
| `Kemampuan Penalaran Umum`        | Nilai UTBK Kemampuan Penalaran Umum                                | `int64`        | `float64` (setelah scaling)         |
| `Pengetahuan dan Pemahaman Umum`  | Nilai UTBK Pengetahuan dan Pemahaman Umum                          | `int64`        | `float64` (setelah scaling)         |
| `Pengetahuan Kuantitatif`         | Nilai UTBK Pengetahuan Kuantitatif                                 | `int64`        | `float64` (setelah scaling)         |
| `Sejarah`                         | Nilai UTBK mata pelajaran Sejarah                                  | `int64`        | `float64` (setelah scaling)         |
| `Sosiologi`                       | Nilai UTBK mata pelajaran Sosiologi                                | `int64`        | `float64` (setelah scaling)         |
| `Total`                           | Total/rata-rata nilai dari semua subjek kemampuan (fitur penting)  | `float64`      | `float64` (setelah scaling)         |
| `Keterangan`                      | **Target/Label:** Status kelulusan ('Lolos' atau 'Tidak Lolos')    | `object`       | `int64` (0: Tidak Lolos, 1: Lolos)  |

### 3.4. Exploratory Data Analysis (EDA)

EDA dilakukan untuk mendapatkan wawasan lebih dalam mengenai struktur data, distribusi fitur, dan hubungan antar variabel.

* **Distribusi Kelas Target:** Visualisasi *countplot* menunjukkan dominasi kelas 'Lolos', menggarisbawahi perlunya metrik evaluasi yang tepat untuk *class imbalance*.
    ![Distribusi Kelas Target](https://github.com/NamaUserAnda/NamaRepoAnda/blob/main/images/countplot_keterangan.png?raw=true)
* **Distribusi Skor Mata Pelajaran:** *Histogram* untuk setiap mata pelajaran menunjukkan sebaran skor, dengan sebagian besar terdistribusi mendekati normal.
    ![Distribusi Skor Mata Pelajaran](https://github.com/NamaUserAnda/NamaRepoAnda/blob/main/images/histplots_scores.png?raw=true)
* **Analisis Korelasi:** *Heatmap* korelasi menunjukkan bahwa **kolom 'Total' memiliki korelasi positif yang sangat tinggi dengan 'Keterangan' (status kelulusan)**. Hal ini menunjukkan bahwa total skor merupakan indikator kelulusan yang paling kuat. Fitur-fitur lain juga berkorelasi positif namun dengan kekuatan yang bervariasi.
    ![Heatmap Korelasi](https://github.com/NamaUserAnda/NamaRepoAnda/blob/main/images/correlation_heatmap.png?raw=true)
* **Deteksi Outlier:** *Box plot* digunakan untuk mengidentifikasi keberadaan *outlier* pada fitur-fitur skor. Meskipun ada beberapa *outlier*, jumlahnya tidak signifikan dan tidak memerlukan penanganan khusus karena model *ensemble* cenderung robust terhadapnya.

---

## 4. Data Preparation

Tahap ini melibatkan pembersihan dan transformasi data agar siap untuk proses pemodelan Machine Learning.

### 4.1. Teknik Data Preparation yang Dilakukan

1.  **Pengecekan dan Penanganan Missing Values:**
    * **Proses:** Dilakukan pengecekan `df.isnull().sum()`.
    * **Alasan:** Memastikan integritas data dan menghindari error pada model.
    * **Hasil:** Ditemukan **tidak ada *missing values***, sehingga tidak ada penanganan lebih lanjut yang diperlukan.
2.  **Pengecekan dan Penanganan Duplicated Rows:**
    * **Proses:** Dilakukan pengecekan `df.duplicated().sum()`.
    * **Alasan:** Baris duplikat dapat menyebabkan *bias* pada model dan menggelembungkan ukuran dataset.
    * **Hasil:** Ditemukan **tidak ada baris duplikat**, sehingga tidak ada penghapusan baris duplikat yang diperlukan.
3.  **Penanganan Kolom Tidak Relevan:**
    * **Proses:** Kolom `absen`, `NISN`, dan `NPSN` dihapus dari dataset.
    * **Alasan:** Kolom-kolom ini merupakan *identifier* unik yang tidak memiliki nilai prediktif untuk status kelulusan. Menghapusnya mengurangi dimensi data dan membantu model fokus pada fitur yang relevan.
    * **Kode Snippet (Contoh):**
        ```python
        columns_to_drop = ['absen', 'NISN', 'NPSN']
        df_cleaned = df.drop(columns=columns_to_drop, axis=1)
        ```
4.  **Encoding Kolom Target 'Keterangan':**
    * **Proses:** Kolom `Keterangan` yang bertipe `object` ('Lolos', 'Tidak Lolos') diubah menjadi numerik (1, 0) menggunakan `sklearn.preprocessing.LabelEncoder`.
    * **Alasan:** Sebagian besar algoritma Machine Learning memerlukan input numerik. Encoding ini memungkinkan model untuk memproses variabel target.
    * **Kode Snippet (Contoh):**
        ```python
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_cleaned['Keterangan'] = le.fit_transform(df_cleaned['Keterangan'])
        # Mapping: 'Lolos' -> 1, 'Tidak Lolos' -> 0 (sesuai hasil fit_transform, biasanya berdasarkan abjad)
        ```
5.  **Pembagian Data (Train-Test Split):**
    * **Proses:** Dataset dibagi menjadi *training set* (80%) dan *testing set* (20%) menggunakan `train_test_split` dari Scikit-learn. Parameter `stratify=y` digunakan.
    * **Alasan:** Memastikan bahwa model dievaluasi pada data yang belum pernah dilihat sebelumnya (data uji) untuk mengukur kemampuan generalisasinya. Penggunaan `stratify=y` sangat penting untuk mempertahankan proporsi kelas target (Lolos/Tidak Lolos) yang sama di kedua set, terutama karena adanya *class imbalance*.
    * **Kode Snippet (Contoh):**
        ```python
        from sklearn.model_selection import train_test_split
        X = df_cleaned.drop('Keterangan', axis=1)
        y = df_cleaned['Keterangan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        ```
6.  **Feature Scaling (Standardisasi):**
    * **Proses:** Fitur-fitur numerik (semua kolom selain target) distandardisasi menggunakan `StandardScaler` dari Scikit-learn. Scaler dilatih pada data pelatihan (`fit_transform`) dan kemudian diterapkan pada data uji (`transform`).
    * **Alasan:** Algoritma Machine Learning tertentu (seperti Logistic Regression atau model berbasis jarak) sensitif terhadap skala fitur. Standardisasi memastikan semua fitur memiliki rata-rata nol dan standar deviasi satu, mencegah fitur dengan nilai yang lebih besar mendominasi proses pembelajaran model.
    * **Kode Snippet (Contoh):**
        ```python
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ```

---

## 5. Modeling

Pada tahap ini, tiga model klasifikasi diimplementasikan untuk memecahkan masalah prediksi kelulusan UTBK.

### 5.1. Logistic Regression

* **Deskripsi Algoritma:** Logistic Regression adalah algoritma klasifikasi linier yang menggunakan fungsi logistik untuk memodelkan probabilitas suatu kelas. Meskipun namanya mengandung "regression", ini adalah model klasifikasi biner yang fundamental.
* **Kelebihan:** Sederhana, cepat dilatih, dan mudah diinterpretasikan karena hubungannya linier. Bekerja dengan baik sebagai *baseline* model.
* **Kekurangan:** Memiliki asumsi linieritas antara fitur dan log-odds dari variabel target, sehingga mungkin tidak berkinerja optimal pada data yang kompleks atau memiliki hubungan non-linier. Rentan terhadap *outlier* dan *multicollinearity*.
* **Tahapan & Parameter:**
    * Model dilatih menggunakan kelas `LogisticRegression` dari `sklearn.linear_model`.
    * Parameter `class_weight='balanced'` digunakan untuk mengatasi *class imbalance* pada dataset, memberikan bobot yang lebih tinggi pada kelas minoritas selama pelatihan.
    * Parameter `solver='liblinear'` dipilih karena efisiensi dan dukungannya terhadap `class_weight`.
    * **Kode Snippet (Contoh):**
        ```python
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        log_reg.fit(X_train_scaled, y_train)
        ```

### 5.2. Random Forest Classifier

* **Deskripsi Algoritma:** Random Forest adalah model *ensemble* berbasis pohon keputusan yang membangun banyak pohon keputusan secara independen selama pelatihan. Hasil prediksi akhir ditentukan oleh *voting* mayoritas dari prediksi masing-masing pohon (untuk klasifikasi). Model ini dirancang untuk mengurangi *overfitting* yang sering terjadi pada pohon keputusan tunggal.
* **Kelebihan:** Sangat *robust* terhadap *overfitting* dan dapat menangani data non-linier serta interaksi kompleks antar fitur. Secara inheren melakukan pemilihan fitur (memberikan *feature importance*). Mampu menangani *missing values* (walaupun kita sudah menanganinya secara eksplisit).
* **Kekurangan:** Kurang interpretabel dibandingkan model linier atau pohon tunggal karena sifat *ensemble*-nya. Bisa membutuhkan waktu pelatihan yang lebih lama dan memori yang lebih besar jika jumlah pohon (`n_estimators`) sangat banyak.
* **Tahapan & Parameter (dengan Improvement: Hyperparameter Tuning):**
    * Model dilatih menggunakan kelas `RandomForestClassifier` dari `sklearn.ensemble`.
    * **Proses Improvement:** Dilakukan *hyperparameter tuning* menggunakan `GridSearchCV`. Ini memungkinkan pencarian kombinasi parameter terbaik dari ruang parameter yang telah ditentukan untuk mengoptimalkan kinerja model.
    * **Parameter yang di-*tuning*:** `n_estimators` (jumlah pohon), `max_features` (jumlah fitur yang dipertimbangkan untuk setiap *split*), `min_samples_leaf` dan `min_samples_split` (kontrol kedalaman dan kompleksitas pohon), serta `class_weight`.
    * **Parameter Optimal (Contoh dari tuning):** `{'class_weight': 'balanced', 'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 150}` (sesuai hasil tuning Anda di *notebook*).
    * **Kode Snippet (Contoh):**
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        param_grid_rf = {
            'n_estimators': [100, 150], # Contoh range yang diuji
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 2],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }
        grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid_search_rf.fit(X_train_scaled, y_train)
        best_rf_model = grid_search_rf.best_estimator_
        ```

### 5.3. XGBoost Classifier

* **Deskripsi Algoritma:** XGBoost (Extreme Gradient Boosting) adalah implementasi *gradient boosting* yang sangat efisien dan populer. Algoritma ini membangun pohon keputusan secara sekuensial, di mana setiap pohon baru belajar untuk mengoreksi kesalahan prediksi dari pohon-pohon sebelumnya.
* **Kelebihan:** Menawarkan performa yang sangat tinggi dan seringkali menjadi *state-of-the-art* dalam kompetisi *machine learning*. Cepat dalam pelatihan dan mampu menangani berbagai jenis data dan pola kompleks. Mendukung penanganan *missing values* secara internal.
* **Kekurangan:** Lebih rentan terhadap *overfitting* jika *hyperparameter* tidak diatur dengan cermat. Proses *hyperparameter tuning* bisa lebih kompleks karena banyaknya parameter yang bisa diatur.
* **Tahapan & Parameter (dengan Improvement: Hyperparameter Tuning):**
    * Model dilatih menggunakan kelas `XGBClassifier` dari pustaka `xgboost`.
    * **Proses Improvement:** Dilakukan *hyperparameter tuning* menggunakan `GridSearchCV` untuk mengidentifikasi kombinasi parameter yang menghasilkan kinerja optimal.
    * **Parameter yang di-*tuning*:** `n_estimators` (jumlah *boosting rounds*), `learning_rate` (ukuran langkah untuk setiap pohon baru), `max_depth` (kedalaman maksimum pohon), `subsample` (proporsi sampel yang digunakan untuk setiap pohon), dan `colsample_bytree` (proporsi fitur yang digunakan untuk setiap pohon).
    * **Parameter Optimal (Contoh dari tuning):** `{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 150, 'subsample': 0.8}` (sesuai hasil tuning Anda di *notebook*).
    * **Kode Snippet (Contoh):**
        ```python
        import xgboost as xgb
        # param_grid_xgb = { ... } # Definisi parameter grid
        grid_search_xgb = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                                     param_grid_xgb, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid_search_xgb.fit(X_train_scaled, y_train)
        best_xgb_model = grid_search_xgb.best_estimator_
        ```

### 5.4. Pemilihan Model Terbaik

Berdasarkan hasil evaluasi (detail akan dijelaskan pada bagian `Evaluation`), model **Random Forest Classifier** dipilih sebagai solusi terbaik untuk proyek ini.

**Alasan pemilihan model terbaik:**
* **Performa Sempurna:** Random Forest mencapai `Accuracy Score` dan `F1-Score` **1.00** pada data uji. Ini menunjukkan kemampuan prediksi yang sempurna untuk kedua kelas ('Lolos' dan 'Tidak Lolos') tanpa *False Positives* maupun *False Negatives*.
* **Robustness terhadap Class Imbalance:** Meskipun dataset memiliki *class imbalance*, model Random Forest yang di-*tuning* dengan `class_weight='balanced'` mampu mengklasifikasikan kelas minoritas ('Tidak Lolos') dengan sempurna.
* **Konsistensi Feature Importance:** Selain performa superior, Random Forest juga memberikan insight yang jelas mengenai fitur paling penting, yaitu `Total` skor, yang konsisten dengan observasi dari EDA dan hasil dari model XGBoost.

---

## 6. Evaluation

Pada bagian ini, kami akan menjelaskan metrik evaluasi yang digunakan dan hasil kinerja model berdasarkan metrik tersebut. Metrik evaluasi yang digunakan telah disesuaikan dengan konteks data (klasifikasi biner dengan *class imbalance*), *problem statement*, dan solusi yang diinginkan.

### 6.1. Metrik Evaluasi yang Digunakan

Untuk mengukur kinerja model klasifikasi, metrik-metrik berikut digunakan:

1.  **Accuracy Score:**
    * **Formula:** $`Accuracy = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Prediksi}}`$
    * **Penjelasan:** Mengukur proporsi total observasi (baik positif maupun negatif) yang diprediksi dengan benar oleh model. Metrik ini memberikan gambaran umum tentang kinerja model, namun bisa menyesatkan pada dataset dengan *class imbalance* karena model mungkin hanya bagus dalam memprediksi kelas mayoritas.
2.  **Confusion Matrix:**
    * **Penjelasan:** Sebuah tabel yang merangkum kinerja model klasifikasi pada sekumpulan data uji yang hasilnya diketahui. Matriks ini menyajikan jumlah True Positives, True Negatives, False Positives, dan False Negatives.
    * **Elemen:**
        * **True Positive (TP):** Jumlah kasus positif yang diprediksi dengan benar oleh model.
        * **True Negative (TN):** Jumlah kasus negatif yang diprediksi dengan benar oleh model.
        * **False Positive (FP):** Jumlah kasus negatif yang salah diprediksi sebagai positif (juga dikenal sebagai Error Tipe I).
        * **False Negative (FN):** Jumlah kasus positif yang salah diprediksi sebagai negatif (juga dikenal sebagai Error Tipe II).
    * **Manfaat:** Memberikan gambaran rinci tentang jenis-jenis kesalahan yang dilakukan model, yang sangat penting untuk memahami kinerja model pada setiap kelas secara individual.
    * **Contoh Visualisasi (dari Logistic Regression):**
        ![Confusion Matrix Logistic Regression](https://github.com/NamaUserAnda/NamaRepoAnda/blob/main/images/confusion_matrix_logreg.png?raw=true)
3.  **Classification Report (Precision, Recall, F1-Score):**
    * **Precision (Presisi):**
        * **Formula:** $`Precision = \frac{\text{TP}}{\text{TP + FP}}`$
        * **Penjelasan:** Mengukur proporsi prediksi positif yang benar-benar positif. Ini menjawab pertanyaan: "Dari semua yang diprediksi positif oleh model, berapa banyak yang sebenarnya positif?" Tinggi presisi berarti model memiliki tingkat *False Positive* yang rendah.
    * **Recall (Sensitivitas / Rekol):**
        * **Formula:** $`Recall = \frac{\text{TP}}{\text{TP + FN}}`$
        * **Penjelasan:** Mengukur proporsi aktual positif yang berhasil diidentifikasi dengan benar oleh model. Ini menjawab pertanyaan: "Dari semua yang sebenarnya positif, berapa banyak yang berhasil ditemukan oleh model?" Tinggi recall berarti model memiliki tingkat *False Negative* yang rendah.
    * **F1-Score:**
        * **Formula:** $`F1-Score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}`$
        * **Penjelasan:** Rata-rata harmonik dari Precision dan Recall. F1-Score adalah metrik yang sangat penting untuk dataset dengan *class imbalance* karena memberikan keseimbangan antara Precision dan Recall, dan memberikan gambaran yang lebih akurat tentang kinerja model daripada akurasi semata.
    * **Support:** Jumlah kemunculan setiap kelas dalam data uji. Untuk proyek ini, *support* kelas 0 ('Tidak Lolos') adalah 1173 dan kelas 1 ('Lolos') adalah 6068, dengan total 7241 sampel uji.

### 6.2. Hasil Proyek Berdasarkan Metrik Evaluasi

Berikut adalah ringkasan kinerja dari setiap model klasifikasi pada data uji (total 7241 sampel):

| Metrik                             | Logistic Regression | Random Forest (Tuned) | XGBoost (Tuned) |
| :--------------------------------- | :------------------ | :-------------------- | :-------------- |
| Accuracy Score                     | 0.9862    | **1.0000** | 0.9981    |
| F1-Score (weighted)                | 0.99      | **1.00** | 1.00      |
| F1-Score (kelas 0 - Tidak Lolos)   | 0.96      | **1.00** | 0.99      |
| F1-Score (kelas 1 - Lolos)         | 0.96      | **1.00** | 1.00      |
| False Positives (FP)               | 90        | **0**   | 7         |
| False Negatives (FN)               | 0         | **0**   | 7         |
| True Positives (TP)                | 5978      | **6068** | 6061      |
| True Negatives (TN)                | 1173      | **1173** | 1166      |

**Analisis Hasil:**

* **Logistic Regression:** Model *baseline* ini menunjukkan *Accuracy Score* **0.9862**. Dari *Confusion Matrix*, terdapat 5978 *True Positives* (peserta 'Lolos' yang diprediksi 'Lolos'), 1173 *True Negatives* (peserta 'Tidak Lolos' yang diprediksi 'Tidak Lolos'), 90 *False Positives* (peserta 'Tidak Lolos' yang salah diprediksi 'Lolos'), dan 0 *False Negatives* (peserta 'Lolos' yang salah diprediksi 'Tidak Lolos'). F1-Score untuk kelas minoritas ('Tidak Lolos') adalah 0.96, dan untuk kelas mayoritas ('Lolos') adalah 0.96. Ini menunjukkan kinerja yang sangat baik untuk model *baseline*.
* **Random Forest Classifier (Tuned):** Model ini mencapai performa **sempurna** pada data uji. Dengan *Accuracy Score* 1.0000 dan *F1-Score* 1.00 untuk kedua kelas (precision, recall, f1-score untuk kelas 0 dan 1 sama-sama 1.00), model ini tidak membuat kesalahan klasifikasi sama sekali (0 *False Positives*, 0 *False Negatives*). Ini menunjukkan kemampuan model yang luar biasa dalam mengklasifikasikan status kelulusan pada dataset ini.
* **XGBoost Classifier (Tuned):** XGBoost juga menunjukkan performa yang sangat tinggi, mendekati sempurna. Dengan *Accuracy Score* 0.9981 dan *F1-Score* 1.00 (weighted average), model ini memberikan kinerja yang sangat kuat. Terdapat sangat sedikit kesalahan klasifikasi (7 *False Positives* dan 7 *False Negatives*), yang masih jauh lebih baik dibandingkan Logistic Regression.

**Keselarasan dengan Tujuan:**
Metrik `F1-Score` digunakan secara khusus karena relevansinya dalam menghadapi *class imbalance* pada dataset target. Model Random Forest yang dipilih berhasil mencapai `F1-Score` sempurna untuk kedua kelas, yang melampaui target awal proyek (`f1_score` di atas 0.90). Dengan demikian, tujuan proyek telah tercapai dengan sangat baik.

---

## 7. Kesimpulan Proyek

Proyek ini berhasil mengembangkan model Machine Learning yang sangat efektif dalam memprediksi status kelulusan peserta UTBK Soshum 2020. Setelah tahap **Data Understanding** mengidentifikasi kualitas data yang baik namun dengan *class imbalance* pada target, dan **Exploratory Data Analysis (EDA)** menyoroti `Total` skor sebagai prediktor paling kuat, model-model klasifikasi dilatih dan dievaluasi.

Model **Random Forest Classifier** yang telah di-*tuning* menunjukkan performa superior dengan **akurasi dan F1-Score sempurna (1.00)** pada data uji (7241 sampel). Ini menandakan bahwa model mampu mengklasifikasikan semua peserta uji dengan benar tanpa kesalahan klasifikasi. Model **XGBoost** juga memberikan kinerja yang sangat kuat, mendekati kesempurnaan dengan *Accuracy Score* 0.9981. Meskipun **Logistic Regression** menunjukkan kinerja yang sangat baik sebagai *baseline* (dengan *Accuracy Score* 0.9862), kedua model *ensemble* unggul dalam mencapai performa yang hampir tanpa cela.

Analisis *feature importance* secara konsisten mengkonfirmasi bahwa `Total` skor dan `Kemampuan Penalaran Umum` adalah fitur paling penting dalam memprediksi kelulusan.

Hasil ini dapat menjadi alat yang sangat berharga bagi institusi pendidikan untuk memprediksi status kelulusan peserta UTBK secara akurat, mendukung pengambilan keputusan, dan memungkinkan intervensi atau perencanaan yang lebih tepat sasaran.

---

## 8. Rekomendasi

Untuk pengembangan dan peningkatan di masa mendatang, beberapa rekomendasi yang dapat dipertimbangkan adalah:

1.  **Validasi Eksternal:** Menguji model pada dataset dari tahun atau gelombang UTBK yang berbeda untuk memastikan generalisasi model yang kuat dan kemampuannya untuk bekerja pada data yang belum pernah dilihat sama sekali.
2.  **Eksplorasi Teknik Penanganan Imbalance Lanjutan:** Meskipun `class_weight='balanced'` bekerja dengan baik, menguji metode *oversampling* (misalnya SMOTE) atau *undersampling* lainnya dapat memberikan pemahaman lebih lanjut tentang bagaimana model bereaksi terhadap penanganan imbalance yang berbeda.
3.  **Feature Engineering Lanjutan:** Mencoba membuat fitur-fitur baru berdasarkan kombinasi atau transformasi dari fitur yang ada (misalnya, rasio skor antar mata pelajaran) yang mungkin dapat menangkap pola yang lebih kompleks.
4.  **Interpretasi Model Mendalam:** Menggunakan alat seperti SHAP (SHapley Additive exPlanations) untuk memberikan penjelasan yang lebih granular tentang bagaimana setiap fitur memengaruhi prediksi untuk setiap individu, yang dapat meningkatkan kepercayaan dan pemahaman *stakeholder*.

---