from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from utils.settings import settings
import logging

def initialize_models():
    """Initialize LLM models for evaluation."""
    logger = logging.getLogger("big_five_eval")
    
    models = {}
    model_versions = {}
    
    logger.info("Initializing models")
    
    if settings.openai_api_key:
        logger.info("Initializing GPT-4o-mini")
        models["GPT-4o-mini"] = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.openai_api_key
        )
        model_versions["GPT-4o-mini"] = "OpenAI API, gpt-4o-mini"
    else:
        logger.warning("OpenAI API key not found, skipping GPT-4o-mini")
    
    if settings.anthropic_api_key:
        logger.info("Initializing Claude 3")
        models["Claude 3"] = ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0,
            api_key=settings.anthropic_api_key
        )
        model_versions["Claude 3"] = "Anthropic API, claude-3-opus-20240229"
    else:
        logger.warning("Anthropic API key not found, skipping Claude 3")
    
    if settings.xai_api_key:
        logger.info("Initializing Grok")
        models["Grok"] = ChatXAI(
            model="grok-beta",
            temperature=0,
            api_key=settings.xai_api_key
        )
        model_versions["Grok"] = "xAI API, grok-beta"
    else:
        logger.warning("xAI API key not found, skipping Grok")
    
    return models, model_versions 