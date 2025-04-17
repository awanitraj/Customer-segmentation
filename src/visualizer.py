import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(data, labels, centers):
    """
    Visualize the clusters with a scatter plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title('Customer Segments')
    plt.xlabel('Age')
    plt.ylabel('Annual Income (k$)')
    plt.legend()
    plt.show()
