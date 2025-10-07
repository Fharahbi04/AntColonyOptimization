# 🐜 Dashboard K-Means & Ant Colony Optimization

Dashboard berbasis **Streamlit** untuk melihat bagaimana hasil pengelompokan menggunakan algoritma **K-Means** yang mana kekurangan dalam memilih centroidnya di atasi dengan 
menggunakan algoritma **Ant Colony Optimization**.

---

## 🚀 Features
- Tampilan interaktif berbasis Streamlit.
- Perbedaan hasil pengelompokkan dengan K-Means dan dengan K-Means & Ant Colony Optimization(ACO).
- Grafik Perbandingan Silhoutte Score dari masing-masing kelompok.


## 📂 Project Structure
```
├── aco.py # Optional Streamlit app (Jika ingin menggunakan fungsi aco yang telah disimpan kedalam aco_model.pkl)
├── AntColonyOptimization.py # main Streamlit app
├── aco_model.pkl # Fungsi ACO yang telah disimpan
├── SKRIPSI2200018132.ipynb # Notebook Analisis
├── requirements.txt # Daftar dependencies
└── data/ # Data asli & Hasil preprocessing
```

## 🛠 Installation
### Clone
   ```bash
   git clone https://github.com/USERNAME/AntColonyOptimization.git
   cd INTERFACEACO
```
### Requirements

```
pip install -r requirements.txt 
```

### Run Streamlit
```
streamlit run AntColonyOptimization.py
```
