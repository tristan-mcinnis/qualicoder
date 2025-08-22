import logging
import os
from datetime import datetime
from termcolor import cprint
from .config import Config

class ColoredLogger:
    """Custom logger with colored output and file logging."""
    
    def __init__(self, name: str = "qualitative_coding"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up file and console handlers."""
        # Ensure logs directory exists
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # File handler
        log_file = os.path.join(Config.LOGS_DIR, f"qualitative_coding_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, color: str = "green"):
        """Log info message with color."""
        cprint(f"✅ {message}", color)
        self.logger.info(message)
    
    def warning(self, message: str, color: str = "yellow"):
        """Log warning message with color."""
        cprint(f"⚠️  {message}", color)
        self.logger.warning(message)
    
    def error(self, message: str, color: str = "red"):
        """Log error message with color."""
        cprint(f"❌ {message}", color)
        self.logger.error(message)
    
    def processing(self, message: str, color: str = "yellow"):
        """Log processing message with color."""
        cprint(f"🔄 {message}", color)
        self.logger.info(message)
    
    def success(self, message: str, color: str = "cyan"):
        """Log success message with color."""
        cprint(f"🚀 {message}", color)
        self.logger.info(message)

# Global logger instance
logger = ColoredLogger()