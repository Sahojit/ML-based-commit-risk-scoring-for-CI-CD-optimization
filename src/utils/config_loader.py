
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigLoader:
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        
        load_dotenv()
        
        logger.info("ConfigLoader initialized")
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {filename}")
        return config
    
    def load_main_config(self) -> Dict[str, Any]:
        return self.load_yaml("config.yaml")
    
    def load_db_config(self) -> Dict[str, Any]:
        config = self.load_yaml("db_config.yaml")
        
        config = self._resolve_env_vars(config)
        
        return config
    
    def _resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_spec = config[2:-1]
            
            if ":" in var_spec:
                var_name, default = var_spec.split(":", 1)
            else:
                var_name = var_spec
                default = None
            
            value = os.getenv(var_name, default)
            
            if value is None:
                raise ValueError(f"Environment variable {var_name} not set and no default provided")
            
            return value
        else:
            return config
    
    def get_database_url(self) -> str:
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
        config = self.load_yaml(config_file)
        
        keys = key.split(".")
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_llm_config(self) -> Dict[str, Any]:
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

if __name__ == "__main__":
    loader = ConfigLoader()
    
    print("=" * 70)
    print("CONFIGURATION LOADER TEST")
    print("=" * 70)
    
    try:
        config = loader.load_main_config()
        print(f"\n✅ Project name: {config['project']['name']}")
        print(f"✅ Project version: {config['project']['version']}")
    except Exception as e:
        print(f"\n❌ Failed to load main config: {e}")
    
    try:
        test_size = loader.get("training.test_size")
        print(f"✅ Test size: {test_size}")
    except Exception as e:
        print(f"❌ Failed to get test_size: {e}")
    
    try:
        db_config = loader.load_db_config()
        print(f"✅ Database host: {db_config['database']['host']}")
        print(f"✅ Database name: {db_config['database']['database']}")
    except Exception as e:
        print(f"⚠️  Database config: {e}")
    
    try:
        llm_config = loader.get_llm_config()
        print(f"\n🤖 LLM Configuration:")
        print(f"   Provider: {llm_config['provider']}")
        print(f"   Model: {llm_config['model']}")
        print(f"   Enabled: {llm_config['enabled']}")
        if llm_config['enabled']:
            print(f"   API Key: {llm_config['api_key'][:20]}... (hidden)")
        else:
            print(f"   ⚠️  API key not configured")
    except Exception as e:
        print(f"❌ Failed to get LLM config: {e}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION LOADER TEST COMPLETE")
    print("=" * 70)