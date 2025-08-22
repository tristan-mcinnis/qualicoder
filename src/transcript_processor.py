"""
Specialized processor for market research transcripts.
Handles focus groups and interviews with proper context preservation.
"""

import re
from typing import List, Dict, Tuple
from .logger import logger

class TranscriptProcessor:
    """Process market research transcripts while preserving conversational context."""
    
    @staticmethod
    def extract_brand_discussions(transcript: str, brand_name: str = None) -> List[Dict]:
        """
        Extract discussion segments that mention the brand or products.
        
        Args:
            transcript: Full transcript text
            brand_name: Optional brand name to focus on
            
        Returns:
            List of relevant discussion segments with context
        """
        segments = []
        
        # Split by speaker turns (common patterns in transcripts)
        speaker_pattern = r'(P\d+.*?\[[\d:]+\s*-\s*[\d:]+\]:|Moderator\s*\[[\d:]+\s*-\s*[\d:]+\]:|Speaker \d+:|Participant \d+:)'
        
        turns = re.split(speaker_pattern, transcript)
        
        # Group speaker and their text
        for i in range(1, len(turns), 2):
            if i+1 < len(turns):
                speaker = turns[i].strip()
                text = turns[i+1].strip()
                
                # Keep segments that are substantial (not just "yeah" or "okay")
                if len(text) > 50:
                    segments.append({
                        'speaker': speaker,
                        'text': text,
                        'length': len(text)
                    })
        
        return segments
    
    @staticmethod
    def create_topic_chunks(transcript: str, chunk_size: int = 5000) -> List[str]:
        """
        Create larger, meaningful chunks that preserve topic discussions.
        
        Args:
            transcript: Full transcript text
            chunk_size: Target size for each chunk (default 5000 chars)
            
        Returns:
            List of topic-based chunks
        """
        chunks = []
        
        # First, try to split by moderator questions (topic boundaries)
        moderator_pattern = r'Moderator\s*\[[\d:]+\s*-\s*[\d:]+\]:'
        
        sections = re.split(moderator_pattern, transcript)
        
        current_chunk = ""
        for section in sections:
            # If adding this section keeps us under chunk_size, add it
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += section + "\n"
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = section
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        # If we ended up with very few chunks, split differently
        if len(chunks) < 3 and len(transcript) > chunk_size * 2:
            # Fall back to splitting by size while preserving sentences
            chunks = TranscriptProcessor._split_by_sentences(transcript, chunk_size)
        
        logger.info(f"Created {len(chunks)} topic-based chunks")
        return chunks
    
    @staticmethod
    def _split_by_sentences(text: str, chunk_size: int) -> List[str]:
        """Split text into chunks while preserving sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def extract_key_themes(transcript: str) -> Dict[str, List[str]]:
        """
        Extract key market research themes from transcript.
        
        Returns:
            Dictionary of theme categories with relevant text segments
        """
        themes = {
            'brand_mentions': [],
            'product_feedback': [],
            'purchase_behavior': [],
            'competitor_mentions': [],
            'price_value': [],
            'usage_context': [],
            'emotional_language': []
        }
        
        # Define patterns for each theme
        patterns = {
            'brand_mentions': r'(Icebreaker|the brand|their products?|they|them)',
            'product_feedback': r'(quality|durability|comfort|fit|fabric|material|design|style|color)',
            'purchase_behavior': r'(buy|bought|purchase|shop|store|online|price|\$|dollar|worth|value)',
            'competitor_mentions': r'(Patagonia|North Face|Kathmandu|Smartwool|REI|other brand)',
            'price_value': r'(expensive|cheap|worth|value|price|cost|afford|budget|money)',
            'usage_context': r'(wear|use|when I|during|for my|hiking|running|travel|work|casual)',
            'emotional_language': r'(love|hate|frustrated|happy|disappointed|excited|annoyed|satisfied)'
        }
        
        # Split into sentences for analysis
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for theme, pattern in patterns.items():
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    themes[theme].append(sentence.strip())
        
        # Remove duplicates and limit to most relevant
        for theme in themes:
            themes[theme] = list(set(themes[theme]))[:10]  # Keep top 10 unique mentions
        
        return themes
    
    @staticmethod
    def prepare_for_coding(transcript: str, max_length: int = 15000) -> str:
        """
        Prepare a focused sample of the transcript for qualitative coding.
        Prioritizes brand discussions and key themes.
        
        Args:
            transcript: Full transcript
            max_length: Maximum characters to send to API
            
        Returns:
            Focused transcript sample
        """
        # Extract key themes
        themes = TranscriptProcessor.extract_key_themes(transcript)
        
        # Build focused sample
        sample = []
        
        # Add some brand mentions
        if themes['brand_mentions']:
            sample.append("=== Brand Discussions ===")
            sample.extend(themes['brand_mentions'][:5])
        
        # Add product feedback
        if themes['product_feedback']:
            sample.append("\n=== Product Feedback ===")
            sample.extend(themes['product_feedback'][:5])
        
        # Add purchase behavior
        if themes['purchase_behavior']:
            sample.append("\n=== Purchase Behavior ===")
            sample.extend(themes['purchase_behavior'][:5])
        
        # Add emotional responses
        if themes['emotional_language']:
            sample.append("\n=== Emotional Responses ===")
            sample.extend(themes['emotional_language'][:5])
        
        # Join and truncate if needed
        focused_sample = "\n".join(sample)
        
        if len(focused_sample) > max_length:
            focused_sample = focused_sample[:max_length]
        
        # If we don't have enough focused content, add general content
        if len(focused_sample) < 1000:
            chunks = TranscriptProcessor.create_topic_chunks(transcript, 5000)
            if chunks:
                focused_sample = "\n".join(chunks[:3])[:max_length]
        
        return focused_sample