# 📘 Judul Proyek
Prediksi Luas Area Kebakaran Hutan Menggunakan Pendekatan Machine Learning dan Deep Learning

## 👤 Informasi
- **Nama:** Mohammad Fakhriza Maftukhin  
- **Repo:** ProyekDataScience  
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi dan menentukan model terbaik  

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
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI MACHINE LEARNING REPOSITORY  
- **Jumlah Data:** 517 
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
- Cleaning (missing/duplicate/outliers)  
- Transformasi (encoding/scaling)  
- Splitting (train/val/test)  

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** [...]  
- **Model 2 – Advanced ML:** [...]  
- **Model 3 – Deep Learning:** [...]  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)

### Hasil Singkat
| Model | Score | Catatan |
|-------|--------|---------|
| Baseline | [...] | |
| Advanced | [...] | |
| Deep Learning | [...] | |

---

# 7. 🏁 Kesimpulan
- Model terbaik: [...]  
- Alasan: [...]  
- Insight penting: [...]  

---

# 8. 🔮 Future Work
- [ ] Tambah data  
- [ ] Tuning model  
- [ ] Coba arsitektur DL lain  
- [ ] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment:
