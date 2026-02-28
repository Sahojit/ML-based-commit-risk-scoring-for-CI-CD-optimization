"""
Configuration Loader Utility
Loads YAML configuration files and environment variables
Supports both OpenAI and Gemini for LLM functionality
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and manages configuration from YAML files and environment variables
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigLoader
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        
        # Load environment variables from .env file
        load_dotenv()
        
        logger.info("ConfigLoader initialized")
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file
        
        Args:
            filename: Name of the YAML file (e.g., 'config.yaml')
        
        Returns:
            Dictionary containing configuration
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {filename}")
        return config
    
    def load_main_config(self) -> Dict[str, Any]:
        """Load main configuration file"""
        return self.load_yaml("config.yaml")
    
    def load_db_config(self) -> Dict[str, Any]:
        """Load database configuration file"""
        config = self.load_yaml("db_config.yaml")
        
        # Replace environment variable placeholders
        config = self._resolve_env_vars(config)
        
        return config
    
    def _resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment variable placeholders in configuration
        
        Format: ${ENV_VAR:default_value}
        """
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # Extract variable name and default value
            var_spec = config[2:-1]  # Remove ${ and }
            
            if ":" in var_spec:
                var_name, default = var_spec.split(":", 1)
            else:
                var_name = var_spec
                default = None
            
            # Get from environment
            value = os.getenv(var_name, default)
            
            if value is None:
                raise ValueError(f"Environment variable {var_name} not set and no default provided")
            
            return value
        else:
            return config
    
    def get_database_url(self) -> str:
        """
        Construct PostgreSQL connection URL
        
        Returns:
            SQLAlchemy connection string
        """
        db_config = self.load_db_config()
        db = db_config['database']
        
        url = (
            f"{db['type']}://"
            f"{db['username']}:{db['password']}"
            f"@{db['host']}:{db['port']}"
            f"/{db['database']}"
        )
        
        return url
    
    def get(self, key: str, config_file: str = "config.yaml", default: Any = None) -> Any:
        """
        Get a specific configuration value
        
        Args:
            key: Dot-separated key path (e.g., 'training.test_size')
            config_file: Configuration file to load
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        config = self.load_yaml(config_file)
        
        # Navigate nested dictionary
        keys = key.split(".")
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration (supports both OpenAI and Gemini)
        
        Returns:
            Dictionary with LLM provider, API key, model, and enabled status
        """
        use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
        
        if use_gemini:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key == "PLACEHOLDER":
                logger.warning("GEMINI_API_KEY not set properly. Explainability features will be disabled.")
                return {
                    "provider": "gemini",
                    "api_key": None,
                    "model": "gemini-1.5-flash",
                    "enabled": False
                }
            return {
                "provider": "gemini",
                "api_key": api_key,
                "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                "enabled": True
            }
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "sk-proj-PLACEHOLDER":
                logger.warning("OPENAI_API_KEY not set properly. Explainability features will be disabled.")
                return {
                    "provider": "openai",
                    "api_key": None,
                    "model": "gpt-4",
                    "enabled": False
                }
            return {
                "provider": "openai",
                "api_key": api_key,
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "enabled": True
            }


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Initialize loader
    loader = ConfigLoader()
    
    print("=" * 70)
    print("CONFIGURATION LOADER TEST")
    print("=" * 70)
    
    # Load main config
    try:
        config = loader.load_main_config()
        print(f"\n✅ Project name: {config['project']['name']}")
        print(f"✅ Project version: {config['project']['version']}")
    except Exception as e:
        print(f"\n❌ Failed to load main config: {e}")
    
    # Get specific value
    try:
        test_size = loader.get("training.test_size")
        print(f"✅ Test size: {test_size}")
    except Exception as e:
        print(f"❌ Failed to get test_size: {e}")
    
    # Load database config (will use placeholders for now)
    try:
        db_config = loader.load_db_config()
        print(f"✅ Database host: {db_config['database']['host']}")
        print(f"✅ Database name: {db_config['database']['database']}")
    except Exception as e:
        print(f"⚠️  Database config: {e}")
    
    # Get LLM config
    try:
        llm_config = loader.get_llm_config()
        print(f"\n🤖 LLM Configuration:")
        print(f"   Provider: {llm_config['provider']}")
        print(f"   Model: {llm_config['model']}")
        print(f"   Enabled: {llm_config['enabled']}")
        if llm_config['enabled']:
            # Only show first 20 chars of API key for security
            print(f"   API Key: {llm_config['api_key'][:20]}... (hidden)")
        else:
            print(f"   ⚠️  API key not configured")
    except Exception as e:
        print(f"❌ Failed to get LLM config: {e}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION LOADER TEST COMPLETE")
    print("=" * 70)