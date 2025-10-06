import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import pickle

# --- Load Dataset (Cache untuk kecepatan) ---
@st.cache_data
def load_data():
    df_final = pd.read_csv("df_final.csv")
    tampilan = pd.read_csv("ukmkuliner.csv")
    return df_final, tampilan

st.set_page_config(page_title="Clustering UMKM Kuliner", layout="wide")
st.title("📊 Clustering UMKM Kuliner dengan K-Means & ACO")

df_final, tampilan = load_data()

st.subheader("📂 Data Asli")
st.write(tampilan.head())
st.subheader("📂 Data Preprocessing Hasil")
st.dataframe(df_final.head())

# ==========================================================
# 🛑 Fungsi untuk memuat ACO dari .pkl (dengan cache)
# ==========================================================
@st.cache_resource
def load_aco_optimizer_cached(filepath, data_ref):
    """Memuat objek AntColonyOptimization dari file .pkl dengan cache."""
    try:
        with open(filepath, 'rb') as f:
            aco_loaded = pickle.load(f)
        aco_loaded.data = data_ref
        st.sidebar.success(f"✅ Optimizer ACO berhasil dimuat dari '{filepath}'")
        return aco_loaded
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memuat ACO dari '{filepath}': {e}")
        # Fallback: Buat instance baru dengan parameter ringan
        return AntColonyOptimization(data_ref, n_clusters=2, n_ants=50, n_iterations=50)

# --- ACO Class  ---
class AntColonyOptimization:
    def __init__(self, data, n_clusters=2, n_ants=50, n_iterations=50, 
                 alpha=1, beta=2, evaporation_rate=0.8):
        self.data = data
        self.n_clusters = n_clusters
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones((len(data), n_clusters))
        
    def optimize(self):
        best_centroids = None
        best_score = -1

        for iteration in range(self.n_iterations):
            all_centroids = []
            all_scores = []

            for _ in range(self.n_ants):
                centroids = self.select_centroids_probabilistic()
                score = self.evaluate_centroids(centroids)
                all_centroids.append(centroids)
                all_scores.append(score)

                if score > best_score:
                    best_centroids = centroids
                    best_score = score

            self.update_pheromone_batch(all_centroids, all_scores)
        
        return best_centroids
    
    def select_centroids_probabilistic(self):
        centroids = []
        remaining_indices = list(range(len(self.data)))

        for cluster_idx in range(self.n_clusters):
            if not remaining_indices:
                chosen_index = np.random.choice(range(len(self.data)))
            else:
                pheromone_values = self.pheromone[remaining_indices, cluster_idx]
                distances = np.ones(len(remaining_indices))

                if centroids:
                    prev_centroids = np.array(centroids)
                    distances = cdist(self.data.iloc[remaining_indices].values, prev_centroids).min(axis=1)
                
                heuristic = 1 / (distances + 1e-6)
                probs = (pheromone_values ** self.alpha) * (heuristic ** self.beta)
                probs /= probs.sum()
                chosen_index = np.random.choice(remaining_indices, p=probs)
                
            centroids.append(self.data.iloc[chosen_index].values)
            if chosen_index in remaining_indices:
                remaining_indices.remove(chosen_index)

        return np.array(centroids)
    
    def evaluate_centroids(self, centroids):
        try:
            kmeans = KMeans(n_clusters=self.n_clusters, init=centroids, n_init=1,
                            max_iter=300, random_state=42)
            kmeans.fit(self.data)
            if len(set(kmeans.labels_)) > 1:
                return silhouette_score(self.data, kmeans.labels_)
            else:
                return -1
        except Exception:
            return -1

    def update_pheromone_batch(self, all_centroids, all_scores):
        self.pheromone *= (1 - self.evaporation_rate)

        for centroids, score in zip(all_centroids, all_scores):
            if score > 0:
                for i, centroid in enumerate(centroids):
                    distances = np.linalg.norm(self.data.values - centroid, axis=1)
                    delta_pheromone = score / (distances + 1e-6)
                    self.pheromone[:, i] += delta_pheromone

        self.pheromone = np.clip(self.pheromone, 1e-6, None)

# ---Centroids---
@st.cache_data
def compute_aco_centroids(k, n_ants, df):
    """Hitung centroid ACO dengan cache berdasarkan k dan n_ants."""
    aco = load_aco_optimizer_cached("aco_model.pkl", df)
    aco.n_clusters = k
    aco.n_ants = n_ants
    with st.spinner(f"Optimasi ACO untuk k={k}... (cached jika tersedia)"):
        return aco.optimize()

# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan Clustering")
n_clusters = st.sidebar.selectbox("Jumlah Cluster (n_clusters)", [2, 3, 4], index=0)
n_ants = st.sidebar.slider("Jumlah Semut (ACO)", 20, 100, 50, step=10) 

# --- K-Means ---
@st.cache_data
def compute_kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(df)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    silhouette = silhouette_score(df, labels)
    return labels, centers, silhouette  # Return tuple: labels, centers, score

st.subheader("1. K-Means ")
kmeans_labels, kmeans_centers, silhouette_kmeans = compute_kmeans(df_final, n_clusters)
st.info(f"K-Means Silhouette Score: {silhouette_kmeans:.4f}")

# --- K-Means + ACO ---
st.subheader("2. K-Means + Ant Colony Optimization (ACO)")
optimal_centroids = compute_aco_centroids(n_clusters, n_ants, df_final)

@st.cache_data
def compute_kmeans_aco(df, centroids, k):
    kmeans_aco = KMeans(n_clusters=k, init=centroids, n_init=1, max_iter=300)
    kmeans_aco.fit(df)
    labels = kmeans_aco.labels_
    centers = kmeans_aco.cluster_centers_
    silhouette = silhouette_score(df, labels)
    return labels, centers, silhouette  # Return tuple: labels, centers, score

kmeans_aco_labels, kmeans_aco_centers, silhouette_aco = compute_kmeans_aco(df_final, optimal_centroids, n_clusters)
st.success(f"K-Means + ACO Silhouette Score: {silhouette_aco:.4f}")

# --- PCA Visualisasi ---
@st.cache_data
def compute_pca_transform(df):
    """Cache hanya PCA transform (cepat dan hashable)."""
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    return df_pca, pca  # Return transformed data dan PCA object

# Fungsi plotting
def plot_clustering(df_pca, kmeans_labels, kmeans_centers, kmeans_sil, 
                    kmeans_aco_labels, kmeans_aco_centers, kmeans_aco_sil, pca):
    """Plot tanpa cache, passing hanya array (hashable)."""
    # Transform centers menggunakan PCA yang sama
    centroids_kmeans_pca = pca.transform(kmeans_centers)
    centroids_aco_pca = pca.transform(kmeans_aco_centers)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # K-Means
    scatter = ax[0].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, cmap="viridis", s=40, alpha=0.7)
    ax[0].scatter(centroids_kmeans_pca[:, 0], centroids_kmeans_pca[:, 1], c="red", marker="X", s=200, label="Centroids")
    ax[0].set_title(f"K-Means (Silhouette={kmeans_sil:.2f})")
    ax[0].legend()
    
    # K-Means + ACO
    scatter = ax[1].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_aco_labels, cmap="plasma", s=40, alpha=0.7)
    ax[1].scatter(centroids_aco_pca[:, 0], centroids_aco_pca[:, 1], c="red", marker="X", s=200, label="Centroids ACO")
    ax[1].set_title(f"K-Means + ACO (Silhouette={kmeans_aco_sil:.2f})")
    ax[1].legend()
    
    plt.colorbar(scatter, ax=ax)  # Opsional: colorbar untuk labels
    return fig

st.subheader(f"📉 Visualisasi Clustering (n_clusters = {n_clusters})")
df_pca, pca = compute_pca_transform(df_final)
fig = plot_clustering(df_pca, kmeans_labels, kmeans_centers, silhouette_kmeans,
                      kmeans_aco_labels, kmeans_aco_centers, silhouette_aco, pca)
st.pyplot(fig)

# --- Perbandingan Silhouette ---
st.subheader("📊 Perbandingan Silhouette Score")
st.write(f"➡️ **K-Means:** {silhouette_kmeans:.4f}")
st.write(f"➡️ **K-Means + ACO:** {silhouette_aco:.4f}")

# --- Analisis Semua Cluster (2,3,4) ---
@st.cache_data
def compute_all_results(n_ants_val, df):
    results = {"Cluster": [], "Metode": [], "Silhouette": []}
    
    for k in [2, 3, 4]:
        # K-Means
        _, _, sil_kmeans = compute_kmeans(df, k)  # Hanya ambil score
        results["Cluster"].append(k)
        results["Metode"].append("K-Means")
        results["Silhouette"].append(sil_kmeans)
        
        # K-Means + ACO
        optimal_centroids_k = compute_aco_centroids(k, n_ants_val, df)
        _, _, sil_aco = compute_kmeans_aco(df, optimal_centroids_k, k)  # Hanya ambil score
        results["Cluster"].append(k)
        results["Metode"].append("K-Means + ACO")
        results["Silhouette"].append(sil_aco)
    
    return pd.DataFrame(results)

st.subheader("📈 Grafik Perbandingan Semua Cluster (2,3,4)")
results_df = compute_all_results(n_ants, df_final)

# --- Plot Grafik ---
@st.cache_data
def plot_comparison(results_df):
    fig2, ax = plt.subplots(figsize=(8, 5))
    for metode in results_df["Metode"].unique():
        subset = results_df[results_df["Metode"] == metode]
        ax.plot(subset["Cluster"], subset["Silhouette"], marker="o", label=metode)
    ax.set_xlabel("Jumlah Cluster (n_clusters)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Perbandingan Silhouette Score")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    return fig2

fig2 = plot_comparison(results_df)
st.pyplot(fig2)

# --- Kesimpulan ---
st.title("Kesimpulan")
st.markdown("""
<style>
.kesimpulan {
    background-color: #f8f9fa;
    border-left: 6px solid #4CAF50;
    padding: 20px;
    border-radius: 10px;
    font-size: 16px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='kesimpulan'>
<h2>📘 Kesimpulan</h2>

🧩 **Kesimpulan Umum:**  
Berdasarkan hasil perbandingan dan visualisasi, didapatkan bahwa **2 cluster memberikan hasil terbaik** dengan akurasi yang baik.

---
            
<h2>📘 Keterangan Cluster</h2>
Ciri-ciri utama dari masing-masing cluster adalah sebagai berikut:
            

💰 **Pendapatan:**  
- Cluster 1: < 10 juta – 70 juta / tahun  
- Cluster 2: 70 juta – >150 juta / tahun  
            
💻 **Pemanfaatan Teknologi:**  
- Cluster 2 unggul 2% dalam promosi digital dibanding Cluster 1 
        
🎓 **Pendidikan:**  
- Cluster 1: didominasi lulusan SMA (pendidikan merata)  
- Cluster 2: didominasi lulusan S1
            
🏠 **Status Kepemilikan Tanah:**  
- Cluster 1: 48% bukan milik sendiri, 52% milik sendiri  
- Cluster 2: 42% bukan milik sendiri, 58% milik sendiri  

🤝 **Bantuan Pemerintah:**  
- Cluster 1: 33% menerima bantuan, 67% tidak  
- Cluster 2: 20% menerima bantuan, 80% tidak  

 
❤️‍🩹 **Asuransi Kesehatan:**
- Cluster 1: 70% anggotanya memiliki asuransi
- Cluster 2: 68% anggotanya memiliki asuransi 
            
🚻 **Jenis Kelamin:**  
- Perempuan lebih banyak di kedua cluster  
- Cluster 1 memiliki 14% lebih banyak perempuan dibanding Cluster 2
</div>
""", unsafe_allow_html=True)