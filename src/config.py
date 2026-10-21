import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Rutas de datos
DATA_DIR = BASE_DIR / "data"
DATA_OPTIMIZADO_DIR = DATA_DIR / "optimizado"
DATA_OPTIMIZADO_DIR.mkdir(parents=True, exist_ok=True)

# Claves de APIs
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

# Configuración de observabilidad con LangSmith
if LANGSMITH_API_KEY and not LANGSMITH_API_KEY.startswith("YOUR_"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "agente-facturas-botilleria")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Parámetros del modelo de embeddings
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384

# Configuración de modelos de IA
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODELS = ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]