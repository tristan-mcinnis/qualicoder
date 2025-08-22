import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from .logger import logger

class LocalVectorStore:
    """Local in-memory vector storage with similarity search."""
    
    def __init__(self):
        """Initialize the local vector store."""
        self.vectors = []
        self.metadata = []
        self.ids = []
        logger.success("Local vector store initialized")
    
    def add_vectors(self, 
                   vectors: List[List[float]], 
                   ids: List[str], 
                   metadata: List[Dict] = None) -> bool:
        """
        Add vectors to the local store.
        
        Args:
            vectors: List of embedding vectors
            ids: List of unique IDs for each vector
            metadata: Optional metadata for each vector
            
        Returns:
            True if successful
        """
        try:
            for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
                self.vectors.append(vector)
                self.ids.append(vector_id)
                
                if metadata and i < len(metadata):
                    self.metadata.append(metadata[i])
                else:
                    self.metadata.append({})
            
            logger.success(f"Added {len(vectors)} vectors to local store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors: {str(e)}")
            return False
    
    def search_similar(self, 
                      query_vector: List[float], 
                      top_k: int = 5,
                      threshold: float = 0.0) -> List[Dict]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Vector to search for
            top_k: Number of similar vectors to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar items with scores
        """
        try:
            if not self.vectors:
                logger.warning("No vectors in store to search")
                return []
            
            # Convert to numpy arrays
            query = np.array(query_vector).reshape(1, -1)
            stored = np.array(self.vectors)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query, stored)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            results = []
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    results.append({
                        'id': self.ids[idx],
                        'score': float(similarities[idx]),
                        'metadata': self.metadata[idx]
                    })
            
            logger.success(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return []
    
    def get_clusters(self, n_clusters: int = 3) -> Dict[int, List[int]]:
        """
        Cluster vectors using K-means.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster IDs to vector indices
        """
        try:
            if not self.vectors:
                return {}
            
            from sklearn.cluster import KMeans
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.vectors)
            
            # Group by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            
            logger.success(f"Created {n_clusters} clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering vectors: {str(e)}")
            return {}
    
    def clear(self):
        """Clear all vectors from the store."""
        self.vectors = []
        self.metadata = []
        self.ids = []
        logger.info("Local vector store cleared")
    
    def size(self) -> int:
        """Get the number of vectors in the store."""
        return len(self.vectors)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if not self.vectors:
            return {'size': 0}
        
        vectors_array = np.array(self.vectors)
        return {
            'size': len(self.vectors),
            'dimensions': vectors_array.shape[1],
            'mean_magnitude': float(np.mean(np.linalg.norm(vectors_array, axis=1))),
            'unique_ids': len(set(self.ids))
        }