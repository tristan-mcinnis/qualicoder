import re
from typing import List
from .logger import logger

class TextPreprocessor:
    """Text preprocessing for qualitative coding analysis."""
    
    @staticmethod
    def preprocess_text(text: str, language: str = 'en') -> str:
        """
        Preprocesses text to remove filler words, noise, and irrelevant content.
        Supports English, Chinese, and Korean.
        
        Args:
            text: Input text to preprocess
            language: Language code ('en', 'zh', 'ko')
            
        Returns:
            Cleaned and preprocessed text
        """
        try:
            # Remove common filler words for English
            if language == 'en':
                text = re.sub(
                    r'\b(uh|um|like|you know|so|actually|basically|literally)\b', 
                    '', 
                    text, 
                    flags=re.IGNORECASE
                )
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Clean specific characters for Chinese and Korean
            if language == 'zh':
                # Keep Chinese characters and basic punctuation
                text = re.sub(r'[^\u4e00-\u9fff\s\.,!?;:]', '', text)
            elif language == 'ko':
                # Keep Korean characters and basic punctuation
                text = re.sub(r'[^\uac00-\ud7af\s\.,!?;:]', '', text)
            
            logger.info(f"Successfully preprocessed text for language: {language}")
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    @classmethod
    def preprocess_texts(cls, texts: List[str], languages: List[str]) -> List[str]:
        """
        Preprocess multiple texts with their corresponding languages.
        
        Args:
            texts: List of texts to preprocess
            languages: List of language codes for each text
            
        Returns:
            List of preprocessed texts
        """
        if len(texts) != len(languages):
            logger.warning("Texts and languages lists have different lengths. Using 'en' for missing languages.")
            languages = languages + ['en'] * (len(texts) - len(languages))
        
        preprocessed_texts = []
        for text, language in zip(texts, languages):
            preprocessed_text = cls.preprocess_text(text, language)
            preprocessed_texts.append(preprocessed_text)
        
        logger.info(f"Successfully preprocessed {len(texts)} texts")
        return preprocessed_texts