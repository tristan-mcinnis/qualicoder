import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for the qualitative coding system."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    PINECONE_API_KEY: Optional[str] = os.getenv('PINECONE_API_KEY')
    PINECONE_HOST: Optional[str] = os.getenv('PINECONE_HOST')
    PINECONE_INDEX_NAME: str = os.getenv('PINECONE_INDEX_NAME', 'qualitative-coding-index')
    PINECONE_DIMENSIONS: int = int(os.getenv('PINECONE_DIMENSIONS', '1536'))
    HUGGING_FACE_TOKEN: Optional[str] = os.getenv('HUGGING_FACE_TOKEN')
    
    # Processing Configuration
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '512'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '128'))
    MIN_SENTENCES_PER_CHUNK: int = int(os.getenv('MIN_SENTENCES_PER_CHUNK', '1'))
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'nomic-ai/modernbert-embed-base')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4o')
    
    # Directory Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUTS_DIR: str = os.path.join(BASE_DIR, 'inputs')
    OUTPUTS_DIR: str = os.path.join(BASE_DIR, 'outputs')
    LOGS_DIR: str = os.path.join(BASE_DIR, 'logs')
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        required_keys = [
            'OPENAI_API_KEY'
        ]
        
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        # Warn about optional keys
        if not cls.HUGGING_FACE_TOKEN:
            print("Warning: HUGGING_FACE_TOKEN not set. Embeddings/similarity search will be disabled.")
        
        return True