import os
import logging
import colorlog
from datetime import datetime

def setup_logger():
    """Set up and configure the logger."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger
    log_file = f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set up color formatter for console
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
            'AI_RESPONSE': 'green',  # Custom level for AI responses
        },
        secondary_log_colors={},
        style='%'
    )
    
    # Set up regular formatter for file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Set up logger
    logger = logging.getLogger("big_five_eval")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Add custom log level for AI responses
    logging.AI_RESPONSE = 25  # Between INFO and WARNING
    logging.addLevelName(logging.AI_RESPONSE, 'AI_RESPONSE')
    
    def ai_response(self, message, *args, **kwargs):
        self.log(logging.AI_RESPONSE, message, *args, **kwargs)
    
    logging.Logger.ai_response = ai_response
    
    logger.info("Logger initialized")
    return logger 