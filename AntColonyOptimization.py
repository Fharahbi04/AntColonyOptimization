import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# --- Load Dataset ---
st.set_page_config(page_title="Clustering UMKM Kuliner", layout="wide")
st.title("📊 Clustering UMKM Kuliner dengan K-Means & ACO")

df_final = pd.read_csv("df_final.csv")
tampilan = pd.read_csv("ukmkuliner.csv")

st.subheader("📂 Data Asli")
st.write(tampilan.head())

st.subheader("📂 Data Preprocessing (Hasil)")
st.dataframe(df_final.head())

# --- ACO Class ---
class AntColonyOptimization:
    def __init__(self, data, n_clusters=2, n_ants=100, n_iterations=100, alpha=1, beta=2, evaporation_rate=0.8):
        self.data = data
        self.n_clusters = n_clusters
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones((len(data), self.n_clusters))

    def optimize(self):
        best_centroids = None
        best_score = -1
        for _ in range(self.n_iterations):
            centroids = self.select_centroids()
            score = self.evaluate_centroids(centroids)
            if score > best_score:
                best_centroids = centroids
                best_score = score
            self.update_pheromone(centroids, score)
        return best_centroids

    def select_centroids(self):
        first_index = np.random.choice(len(self.data))
        centroids = [self.data.iloc[first_index].values]
        for _ in range(1, self.n_clusters):
            dists = cdist(self.data.values, np.array(centroids)).min(axis=1)
            next_index = np.argmax(dists)
            centroids.append(self.data.iloc[next_index].values)
        return np.array(centroids).reshape(self.n_clusters, -1)

    def evaluate_centroids(self, centroids):
        kmeans = KMeans(n_clusters=self.n_clusters, init=centroids, n_init=1, max_iter=300, random_state=42)
        kmeans.fit(self.data)
        if len(set(kmeans.labels_)) > 1:
            return silhouette_score(self.data, kmeans.labels_)
        else:
            return -1

    def update_pheromone(self, centroids, score):
        for i, centroid in enumerate(centroids):
            distances = np.linalg.norm(self.data.values - centroid, axis=1)
            self.pheromone[:, i] *= (1 - self.evaporation_rate)
            self.pheromone[:, i] += score / (distances + 1e-6)
            self.pheromone[:, i] = np.clip(self.pheromone[:, i], 1e-6, None)

# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan Clustering")
n_clusters = st.sidebar.selectbox("Jumlah Cluster (n_clusters)", [2, 3, 4], index=0)
n_ants = st.sidebar.slider("Jumlah Semut (ACO)", 50, 200, 80, step=10)
n_iter = st.sidebar.slider("Jumlah Iterasi (ACO)", 50, 200, 100, step=10)

# --- Cache hasil ACO agar cepat ---
@st.cache_resource
def run_aco(data, n_clusters, n_ants, n_iter):
    aco = AntColonyOptimization(data, n_clusters=n_clusters, n_ants=n_ants, n_iterations=n_iter)
    optimal_centroids = aco.optimize()
    return optimal_centroids

# --- K-Means ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(df_final)
silhouette_kmeans = silhouette_score(df_final, kmeans.labels_)

# --- K-Means + ACO ---
optimal_centroids = run_aco(df_final, n_clusters, n_ants, n_iter)
kmeans_aco = KMeans(n_clusters=n_clusters, init=optimal_centroids, n_init=1, max_iter=300)
kmeans_aco.fit(df_final)
silhouette_aco = silhouette_score(df_final, kmeans_aco.labels_)

# --- PCA Visualisasi ---
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_final)
centroids_pca = pca.transform(kmeans.cluster_centers_)
centroids_aco_pca = pca.transform(kmeans_aco.cluster_centers_)

# --- Visualisasi Clustering ---
st.subheader(f"📉 Visualisasi Clustering (n_clusters = {n_clusters})")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# K-Means
ax[0].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, cmap="viridis", s=40, alpha=0.7)
ax[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="red", marker="X", s=200, label="Centroids")
ax[0].set_title(f"K-Means (Silhouette={silhouette_kmeans:.2f})")
ax[0].legend()

# K-Means + ACO
ax[1].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_aco.labels_, cmap="plasma", s=40, alpha=0.7)
ax[1].scatter(centroids_aco_pca[:, 0], centroids_aco_pca[:, 1], c="red", marker="X", s=200, label="Centroids ACO")
ax[1].set_title(f"K-Means + ACO (Silhouette={silhouette_aco:.2f})")
ax[1].legend()

st.pyplot(fig)

# --- Perbandingan Silhouette ---
st.subheader("📊 Perbandingan Silhouette Score")
st.write(f"➡️ **K-Means:** {silhouette_kmeans:.4f}")
st.write(f"➡️ **K-Means + ACO:** {silhouette_aco:.4f}")

# --- Grafik Perbandingan Line ---
results_df = pd.DataFrame({
    "Cluster": [2, 3, 4, 2, 3, 4],
    "Metode": ["K-Means", "K-Means", "K-Means", "K-Means + ACO", "K-Means + ACO", "K-Means + ACO"],
    "Silhouette": [0.88, 0.81, 0.81, 0.89, 0.86, 0.86]
})

sns.set(style="whitegrid")
fig2, ax = plt.subplots(figsize=(8, 5))

colors = {"K-Means": "#1f77b4", "K-Means + ACO": "#ff7f0e"}

for metode in results_df["Metode"].unique():
    subset = results_df[results_df["Metode"] == metode]
    ax.plot(subset["Cluster"], subset["Silhouette"], marker="o", linewidth=2.5, label=metode, color=colors.get(metode))

ax.set_xlabel("Jumlah Cluster (n_clusters)", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
ax.set_title("📈 Perbandingan Silhouette Score Antara K-Means dan ACO", fontsize=14, fontweight='bold')
ax.legend(title="Metode", fontsize=10, title_fontsize=11)
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax.set_facecolor("#f9f9f9")
ax.set_ylim(0.7, 1.0)
sns.despine()

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
