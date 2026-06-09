from .env import env_str


EXECUTION_DB_PATH = env_str("EXECUTION_DB_PATH")
EXECUTION_DB_TYPE = str(env_str("EXECUTION_DB_TYPE", "sqlite")).lower()

SUPABASE_URL = env_str("SUPABASE_URL", "https://jjuatlyxubeglxkrpaji.supabase.co")
SUPABASE_KEY = env_str("SUPABASE_KEY") or env_str("SUPABASE_SERVICE_KEY")
