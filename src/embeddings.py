import torch
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer, AutoModel
from .config import Config
from .logger import logger

class EmbeddingGenerator:
    """Generate embeddings using HuggingFace models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the tokenizer and model."""
        try:
            logger.processing("Initializing tokenizer and model...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_auth_token=Config.HUGGING_FACE_TOKEN
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                use_auth_token=Config.HUGGING_FACE_TOKEN
            )
            
            logger.success(f"Successfully initialized model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Perform mean pooling on model output.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Mean-pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not self.tokenizer or not self.model:
                raise ValueError("Model not initialized properly")
            
            logger.processing(f"Generating embeddings for {len(texts)} texts...")
            
            # Tokenize texts
            encoded_texts = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded_texts)
            
            # Apply mean pooling
            embeddings = self.mean_pooling(outputs, encoded_texts["attention_mask"])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Convert to list format
            embeddings_list = embeddings.tolist()
            
            logger.success(f"Successfully generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
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
            return 768  # Default dimension for many models