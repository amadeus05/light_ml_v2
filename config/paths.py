from pathlib import Path


MODELS_DIR = Path("models")
BACKTEST_CHARTS_DIR = Path("backtest_charts")

MODELS_DIR.mkdir(exist_ok=True)
BACKTEST_CHARTS_DIR.mkdir(exist_ok=True)
