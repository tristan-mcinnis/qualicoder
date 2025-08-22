from pinecone import Pinecone
from typing import List, Dict, Optional
from .config import Config
from .logger import logger

class PineconeVectorStore:
    """Pinecone vector database integration for storing and retrieving embeddings."""
    
    def __init__(self, index_name: str = None):
        """
        Initialize Pinecone vector store.
        
        Args:
            index_name: Name of the Pinecone index to use
        """
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.index = None
        self.pc = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection and index."""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            
            logger.processing("Initializing Pinecone connection...")
            
            # Connect to or create index
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            
            if self.index_name not in index_names:
                logger.warning(f"Index '{self.index_name}' does not exist. Creating it with 1536 dimensions...")
                self.create_index_for_openai_embeddings()
            else:
                # Check if existing index has correct dimensions
                try:
                    self.index = self.pc.Index(self.index_name, host=Config.PINECONE_HOST)
                    # Test with a dummy vector to check dimensions
                    test_vector = [0.0] * 1536
                    self.index.query(vector=test_vector, top_k=1, include_metadata=False)
                    logger.success("Existing index dimensions are compatible")
                except Exception as e:
                    if "dimension" in str(e).lower():
                        logger.warning(f"Existing index has wrong dimensions. Creating new index: {self.index_name}-1536")
                        self.index_name = f"{self.index_name}-1536"
                        if self.index_name not in index_names:
                            self.create_index_for_openai_embeddings()
                        else:
                            self.index = self.pc.Index(self.index_name, host=Config.PINECONE_HOST)
                    else:
                        raise e
            
            if not self.index:
                self.index = self.pc.Index(self.index_name, host=Config.PINECONE_HOST)
            logger.success(f"Successfully connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def create_index(self, dimension: int, metric: str = "cosine"):
        """
        Create a new Pinecone index.
        
        Args:
            dimension: Dimension of vectors to be stored
            metric: Distance metric for similarity search
        """
        try:
            logger.processing(f"Creating Pinecone index: {self.index_name}")
            
            from pinecone import ServerlessSpec
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            # Wait for index to be ready
            import time
            time.sleep(60)  # Pinecone needs time to initialize
            
            self.index = self.pc.Index(self.index_name, host=Config.PINECONE_HOST)
            logger.success(f"Successfully created Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {str(e)}")
            raise
    
    def create_index_for_openai_embeddings(self):
        """
        Create a Pinecone index specifically for OpenAI embeddings (1536 dimensions).
        """
        try:
            logger.processing(f"Creating Pinecone index for OpenAI embeddings: {self.index_name}")
            
            from pinecone import ServerlessSpec
            
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            logger.processing("Waiting for index to be ready...")
            # Wait for index to be ready
            import time
            time.sleep(30)  # Reduced wait time
            
            self.index = self.pc.Index(self.index_name, host=Config.PINECONE_HOST)
            logger.success(f"Successfully created OpenAI-compatible Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error creating OpenAI-compatible Pinecone index: {str(e)}")
            raise
    
    def upsert_vectors(self, 
                      vectors: List[List[float]], 
                      ids: List[str], 
                      metadata: List[Dict] = None) -> bool:
        """
        Upsert vectors to Pinecone index.
        
        Args:
            vectors: List of embedding vectors
            ids: List of unique IDs for each vector
            metadata: Optional metadata for each vector
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            logger.processing(f"Upserting {len(vectors)} vectors to Pinecone...")
            
            # Prepare upsert data
            upsert_data = []
            for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
                data_point = {
                    'id': vector_id,
                    'values': vector
                }
                if metadata and i < len(metadata):
                    data_point['metadata'] = metadata[i]
                upsert_data.append(data_point)
            
            # Upsert in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.success(f"Successfully upserted {len(vectors)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False
    
    def query_vectors(self, 
                     query_vector: List[float], 
                     top_k: int = 10, 
                     include_metadata: bool = True) -> Dict:
        """
        Query similar vectors from Pinecone index.
        
        Args:
            query_vector: Vector to search for similar vectors
            top_k: Number of similar vectors to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            Query results from Pinecone
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            logger.processing(f"Querying Pinecone for top {top_k} similar vectors...")
            
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata
            )
            
            logger.success(f"Successfully retrieved {len(results.matches)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            return {}
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors from Pinecone index.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            logger.processing(f"Deleting {len(ids)} vectors from Pinecone...")
            
            self.index.delete(ids=ids)
            logger.success(f"Successfully deleted {len(ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Index statistics
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            stats = self.index.describe_index_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}