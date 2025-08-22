from typing import List
from openai import OpenAI
from .config import Config
from .logger import logger

class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI API."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the OpenAI embedding generator.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.success(f"Successfully initialized OpenAI client for embeddings: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized properly")
            
            logger.processing(f"Generating embeddings for {len(texts)} texts using {self.model_name}...")
            
            # Generate embeddings using OpenAI API
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            # Extract embedding vectors
            embeddings = [embedding.embedding for embedding in response.data]
            
            logger.success(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        try:
            # Generate a test embedding
            test_embedding = self.generate_embeddings(["test"])
            return len(test_embedding[0])
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {str(e)}")
            # Default dimensions for common OpenAI models
            if "3-small" in self.model_name:
                return 1536
            elif "3-large" in self.model_name:
                return 3072
            else:
                return 1536  # Default