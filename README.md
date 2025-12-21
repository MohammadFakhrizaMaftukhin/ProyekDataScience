# 📘 Judul Proyek
Prediksi Luas Area Kebakaran Hutan Menggunakan Pendekatan Machine Learning dan Deep Learning

## 👤 Informasi
- **Nama:** Mohammad Fakhriza Maftukhin  
- **Repo:** ProyekDataScience  
- **Video:** Video Pembahasan UAS Data Science

---

# 1. 🎯 Ringkasan Proyek
- Membangun model regresi untuk mengestimasi dampak kebakaran hutan, mengingat karakteristik data yang sangat noisy dan memiliki distribusi skewed.
- Menangani outliers dan distribusi target yang timpang menggunakan Log Transformation.
- Melakukan *Feature Engineering* (membuat fitur `is_weekend`).
- Menerapkan Standard Scaling dan Manual Encoding untuk fitur temporal.
- Membangun 3 Model Machine Learning:
  - Baseline: Linear Regression (Pendekatan Statistik).
  - Advanced: Random Forest Regressor (Pendekatan Ensemble).
  - Deep Learning: Multi-Layer Perceptron / MLP (Pendekatan Neural Network).
- Melakukan komparasi kinerja menggunakan metrik RMSE dan MAE, dengan hasil temuan bahwa model sederhana (Linear Regression) justru mengungguli model kompleks karena keterbatasan jumlah data.

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Model perlu mampu memprediksi estimasi luas area hutan yang terbakar (area) dengan tingkat error seminimal mungkin berdasarkan data meteorologi harian. 
- Dataset memiliki distribusi target yang sangat miring (highly skewed) ke arah nol, sehingga memerlukan strategi preprocessing khusus (seperti transformasi logaritma) agar model tidak bias.
- Diperlukan evaluasi untuk menentukan apakah model kompleks seperti Deep Learning (MLP) mampu memberikan performa yang lebih baik dibandingkan metode Machine Learning konvensional (Random Forest) pada dataset dengan jumlah sampel terbatas.

**Goals:**  
- Membangun model regresi yang mampu memprediksi estimasi luas area kebakaran hutan (area) dengan tingkat kesalahan (error) seminimal mungkin, diukur menggunakan metrik RMSE (Root Mean Squared Error) dan MAE (Mean Absolute Error).  
- Mengukur dan membandingkan performa dari tiga pendekatan algoritma yang berbeda, yaitu Linear Regression (Baseline), Random Forest (Advanced Machine Learning), dan Multilayer Perceptron (Deep Learning), khususnya pada data yang telah melalui transformasi logaritma.
- Menentukan model terbaik yang paling efektif dan robust (kokoh) dalam menangani pola data cuaca yang kompleks dan distribusi target yang miring (skewed).
- Menghasilkan kode eksperimen yang terstruktur dan dapat dijalankan ulang (reproducible) yang terdokumentasi dalam repositori GitHub. 

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
|   ├── model_mlp.h5
│   └── scaler.pkl
│
├── images/                 # Visualizations
|   ├── evaluasi_model1.png
|   ├── evaluasi_model2.png
|   ├── evaluasi_model3.png
|   ├── feature_importance.png
|   ├── model_comparison.png
|   ├── training_loss.png
|   ├── training_mae.png
|   ├── vis1_distribusi.png
|   ├── vis2_heatmap.png
│   └── vis3_scatter.png
│
├── requirements.txt        # Dependencies
├── .gitignore
├── Cheklist Submit.md
├── LICENSE
├── Laporan Proyek Machine Learning.md
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/forest+fires)
- **Jumlah Data:** 517 baris
- **Tipe:** Tabular 

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
| X | koordinat spasial sumbu X dalam peta taman Montesinho |
| Y | koordinat spasial sumbu Y dalam peta taman Montesinho |
| month | bulan dalam setahun |
| day | hari dalam seminggu |
| FFMC | Indeks FFMC dari sistem FWI |
| DMC | Indeks DMC dari sistem FWI |
| DC | Indeks DC dari sistem FWI |
| ISI | Indeks ISI dari sistem FWI |
| temp | suhu (celcius) |
| RH | kelembaban relatif |
| wind | kecepatan angin |
| rain | hujan |

---

# 4. 🔧 Data Preparation
- Cleaning: Menghapus data duplikat, Handling Outliers dengan menerapkan Transformasi Logaritma (Log(x+1)) pada variabel target `area`, Mengubah tipe data kolom month dan day.
- Feature engineering: Membuat fitur baru `is_weekend` dan Memilih 13 fitur prediktor `X`, `Y`, `month`, `day`, `is_weekend`, `FFMC`, `DMC`, `DC`, `ISI`, `temp`, `RH`, `wind`, `rain`.
- Transformasi: Mengimplementasikan ordinal encoding ke kolom `month`, `day` dan Standarisasi fitur numerik menggunakan `StandardScaler`.
- Splitting: Membagi dataset menjadi 80% Training dan 20% Testing dengan `random_state=42`.
- Balancing: Tidak menerapkan karena dataset bersifat regresi.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Linear Regression  
- **Model 2 – Advanced ML:** Random Forest Regressor
- **Model 3 – Deep Learning:** Multi Layer Perceptron

---

# 6. 🧪 Evaluation
**Metrik:** RMSE & MAE

### Hasil Singkat
| Model | RMSE (Log Scale) | MAE (Log Scale) | Catatan |
|-------|------------------|-----------------|---------|
| Baseline (Linear Regression) | 1.2562 | 1.0774 | Paling stabil dan generalisasinya baik. |
| Advanced (Random Forest) | 1.3500 | 1.1573 | Mengalami overfitting (gap tinggi antara train & test). |
| Deep Learning (MLP) | 1.3233 | 1.1153 | Performa moderat, terkendala jumlah data yang sedikit. |

---

# 7. 🏁 Kesimpulan
- Model terbaik: Linear Regression  
- Alasan: Model ini memberikan error terendah pada data uji. Model kompleks seperti Random Forest dan Deep Learning cenderung menangkap noise sebagai pola (overfitting) atau kekurangan data untuk belajar optimal.
- Insight penting: Variabel cuaca (suhu, angin, kelembapan) saja memiliki korelasi yang lemah terhadap luas area kebakaran. Diperlukan data tambahan seperti data geospasial atau vegetasi untuk meningkatkan akurasi.

---

# 8. 🔮 Future Work
- ✅  Mengumpulkan lebih banyak data
- ✅  Menambah variasi data
- ✅  Feature engineering lebih lanjut
- ✅  Hyperparameter tuning lebih ekstensif
- ✅  Membuat web application (Streamlit/Gradio)

---

# 9. 🔁 Reproducibility
Gunakan environment:
Python Version: [3.12]

Cara menjalankan proyek ini:
### 1. Clone Repository
```bash
git clone [https://github.com/MohammadFakhrizaMaftukhin/ProyekDataScience.git](https://github.com/MohammadFakhrizaMaftukhin/ProyekDataScience.git)
cd ProyekDataScience
```

### 2. Setup Environment
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Notebook
```bash
jupyter notebook notebooks/ML_project.ipynb
```
