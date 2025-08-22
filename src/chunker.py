import nltk
from typing import List
from nltk.tokenize import sent_tokenize
from .config import Config
from .logger import logger

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

class TextChunker:
    """Text chunking with sentence boundary preservation."""
    
    def __init__(self, 
                 max_chunk_size: int = None, 
                 overlap_size: int = None, 
                 min_sentences_per_chunk: int = None):
        """
        Initialize TextChunker with configurable parameters.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap_size: Number of characters to overlap between chunks
            min_sentences_per_chunk: Minimum sentences required per chunk
        """
        self.max_chunk_size = max_chunk_size or Config.CHUNK_SIZE
        self.overlap_size = overlap_size or Config.CHUNK_OVERLAP
        self.min_sentences_per_chunk = min_sentences_per_chunk or Config.MIN_SENTENCES_PER_CHUNK
    
    def chunk_text_into_sentences(self, text: str) -> List[str]:
        """
        Chunks text into segments while preserving sentence boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        try:
            # Split text into sentences
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed max_chunk_size
                if current_length + sentence_length > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_point = max(0, len(current_chunk) - self.overlap_size)
                    current_chunk = current_chunk[overlap_point:]
                    current_length = sum(len(s) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            logger.info(f"Successfully split text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in sentence chunking: {str(e)}")
            return [text]  # Return original text if chunking fails
    
    def chunk_texts(self, texts: List[str]) -> List[str]:
        """
        Chunks multiple texts into smaller segments while preserving sentence boundaries.
        
        Args:
            texts: List of texts to chunk
            
        Returns:
            List of all text chunks from all input texts
        """
        try:
            all_chunks = []
            for i, text in enumerate(texts):
                logger.processing(f"Processing text {i+1}/{len(texts)}")
                chunks = self.chunk_text_into_sentences(text)
                
                for chunk in chunks:
                    logger.processing(f"Processing chunk with {len(chunk)} characters")
                    all_chunks.append(chunk)
            
            logger.success(f"Successfully chunked {len(texts)} texts into {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error chunking texts: {str(e)}")
            return texts
    
    def get_chunk_info(self, chunks: List[str]) -> dict:
        """
        Get information about the chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_characters': sum(chunk_lengths)
        }