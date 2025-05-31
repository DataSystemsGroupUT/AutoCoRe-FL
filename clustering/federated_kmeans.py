import logging
import numpy as np
from sklearn.cluster import KMeans

class FederatedKMeans:
    def __init__(self, num_clusters, client_id):
        """
        Initializes the FederatedKMeans instance.
        :param num_clusters: Number of clusters (K) for KMeans.
        :param client_id: Unique identifier for the client.
        """
        self.k = num_clusters
        self.centroids = None
        self.client_id = client_id

    def set_centroids(self, centroids):
        self.centroids = centroids

    def compute_local_stats(self, embeddings):
        """
        Assigns clusters using current centroids, computes local sums and counts for aggregation.
        """
        # Check if centroids are initialized
        logger = logging.getLogger("FederatedKMeans")   
        if self.centroids is None:
            # Initialize with KMeans++ on local data
            logger.info("Initializing KMeans centroids using KMeans++")
            kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            logger.info(f"Client {self.client_id}: KMeans++ centroids initialized.")
            self.centroids = kmeans.cluster_centers_
        else:
            # Assign clusters using provided centroids
            logger.info("Assigning clusters using existing centroids")
            dists = np.linalg.norm(embeddings[:, None, :] - self.centroids[None, :, :], axis=-1)
            labels = np.argmin(dists, axis=1)
        K_live = self.centroids.shape[0]   
        local_sums = np.zeros_like(self.centroids)
        local_counts = np.zeros((K_live,), dtype=np.int32)
        for idx, c in enumerate(labels):
            local_sums[c] += embeddings[idx]
            local_counts[c] += 1

        return local_sums, local_counts, labels