from sklearn.cluster import KMeans

def apply_kmeans(data, n_clusters):
    """
    Apply KMeans clustering on the data and return the model and cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans, kmeans.labels_
