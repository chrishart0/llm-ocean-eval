from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from utils.settings import settings
import logging
import os
import yaml
from pathlib import Path
import importlib

class ModelRegistry:
    """Registry for model configurations and initialization."""
    
    def __init__(self, config_dir="configs"):
        self.logger = logging.getLogger("big_five_eval")
        self.configs = {}
        self.model_instances = {}
        self.api_keys = {
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
            "xai": settings.xai_api_key,
        }
        self._load_configs(config_dir)
    
    def _load_configs(self, config_dir):
        """Load model configurations from YAML files."""
        config_path = Path(config_dir) / "models"
        if not config_path.exists():
            self.logger.warning(f"Config directory {config_path} not found. Using default configurations.")
            self._load_default_configs()
            return
            
        self.logger.info(f"Loading model configurations from {config_path}")
        for file_path in config_path.glob("*.yaml"):
            try:
                with open(file_path, "r") as f:
                    provider_config = yaml.safe_load(f)
                    provider = file_path.stem  # Use filename as provider name
                    self.configs[provider] = provider_config
                    self.logger.info(f"Loaded configuration for provider: {provider}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {file_path}: {str(e)}")
    
    def _load_default_configs(self):
        """Load default configurations when config files are not available."""
        self.configs = {
            "openai": {
                "gpt-4o-mini": {
                    "name": "GPT-4o-mini",
                    "version": "OpenAI API, gpt-4o-mini",
                    "class": "langchain_openai.ChatOpenAI",
                    "params": {
                        "model": "gpt-4o-mini",
                        "temperature": 0
                    }
                },
                "gpt-4o": {
                    "name": "GPT-4o",
                    "version": "OpenAI API, gpt-4o",
                    "class": "langchain_openai.ChatOpenAI",
                    "params": {
                        "model": "gpt-4o",
                        "temperature": 0
                    }
                }
            },
            "anthropic": {
                "claude-3-opus": {
                    "name": "Claude 3 Opus",
                    "version": "Anthropic API, claude-3-opus-20240229",
                    "class": "langchain_anthropic.ChatAnthropic",
                    "params": {
                        "model": "claude-3-opus-20240229",
                        "temperature": 0
                    }
                },
                "claude-3-sonnet": {
                    "name": "Claude 3 Sonnet",
                    "version": "Anthropic API, claude-3-sonnet-20240229",
                    "class": "langchain_anthropic.ChatAnthropic",
                    "params": {
                        "model": "claude-3-sonnet-20240229",
                        "temperature": 0
                    }
                }
            },
            "xai": {
                "grok-beta": {
                    "name": "Grok",
                    "version": "xAI API, grok-beta",
                    "class": "langchain_xai.ChatXAI",
                    "params": {
                        "model": "grok-beta",
                        "temperature": 0
                    }
                }
            }
        }
    
    def _get_class(self, class_path):
        """Dynamically import and return a class from its string path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def list_available_models(self):
        """List all available model configurations."""
        available_models = []
        for provider, models in self.configs.items():
            if not self.api_keys.get(provider):
                continue  # Skip providers without API keys
                
            for model_id, config in models.items():
                available_models.append(f"{provider}:{model_id}")
        
        return available_models
    
    def initialize_models(self, selected_models=None, batch_file=None):
        """
        Initialize models based on selection criteria.
        
        Args:
            selected_models (list, optional): List of specific model identifiers to run.
                                             Format is ["provider:model_id", ...].
            batch_file (str, optional): Path to a YAML file specifying which models to run.
            
        Returns:
            tuple: (models dict, model_versions dict)
        """
        models = {}
        model_versions = {}
        
        # Load batch configuration if specified
        batch_config = []
        if batch_file:
            try:
                with open(batch_file, "r") as f:
                    batch_config = yaml.safe_load(f).get("models", [])
                self.logger.info(f"Loaded batch configuration from {batch_file}")
            except Exception as e:
                self.logger.error(f"Error loading batch configuration: {str(e)}")
                return {}, {}
        
        # Use batch config if provided, else use selected_models if provided, else use all available models
        model_specs = batch_config or selected_models or self.list_available_models()
        
        if isinstance(model_specs, str):
            model_specs = [model_specs]  # Convert single string to list
        
        # For each model to initialize
        for model_spec in model_specs:
            if ":" not in model_spec:
                self.logger.warning(f"Invalid model format: {model_spec}. Should be 'provider:model_id'")
                continue
            
            provider, model_id = model_spec.split(":", 1)
            
            if provider not in self.configs:
                self.logger.warning(f"Unknown provider: {provider}")
                continue
            
            if model_id not in self.configs[provider]:
                self.logger.warning(f"Unknown model ID: {model_id} for provider {provider}")
                continue
            
            if not self.api_keys.get(provider):
                self.logger.warning(f"No API key for {provider}, cannot initialize {model_id}")
                continue
            
            # Check environment variable for enabling/disabling
            env_var = f"ENABLE_{provider}_{model_id}".upper().replace("-", "_")
            if os.environ.get(env_var, "").lower() == "false":
                self.logger.info(f"Skipping {model_id} as specified by {env_var}=false")
                continue
            
            # Initialize the model
            model_config = self.configs[provider][model_id]
            model_name = model_config["name"]
            
            self.logger.info(f"Initializing {model_name}")
            try:
                # Get the model class dynamically
                model_class = self._get_class(model_config["class"])
                
                # Create a copy of params and add API key
                params = model_config["params"].copy()
                params["api_key"] = self.api_keys[provider]
                
                # Initialize the model
                models[model_name] = model_class(**params)
                model_versions[model_name] = model_config["version"]
                self.logger.info(f"Successfully initialized {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {model_name}: {str(e)}")
        
        if not models:
            self.logger.warning("No models were successfully initialized")
        
        return models, model_versions


# Create a backward-compatible function that uses the registry
def initialize_models(selected_model=None, batch_file=None):
    """
    Backward-compatible function for initializing models.
    
    Args:
        selected_model (str, optional): Specific model identifier to run.
        batch_file (str, optional): Path to a YAML file with model selections.
        
    Returns:
        tuple: (models dict, model_versions dict)
    """
    registry = ModelRegistry()
    return registry.initialize_models(
        selected_models=[selected_model] if selected_model else None,
        batch_file=batch_file
    ) 