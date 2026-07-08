# Compatibility shim: re-export from shared config so existing imports still work.
# New code should import directly from shared.config.
from shared.config.settings import CONFIG, LLMConfig, AlpacaConfig, RiskConfig, BacktestConfig, AppConfig
from shared.config.llm import get_llm
from shared.config.json_utils import parse_json_loose
