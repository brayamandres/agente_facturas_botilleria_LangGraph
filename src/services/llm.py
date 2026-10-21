from langchain_openai import ChatOpenAI
from src.config import GROQ_API_KEY, OPENAI_API_KEY, DEFAULT_GROQ_MODEL

def get_reasoning_llm(temperature = 0.0):
    """
    Retorna la instancia del modelo de lenguaje para razonamiento y toma de decisiones.
    Prioriza Groq por velocidad/costo o OpenAI si está disponible.
    
    Args:
        temperature (float): Controla la aleatoriedad de las respuestas del modelo. Default es 0.0 (determinístico).
    
    Returns:
        ChatOpenAI: Instancia del modelo de lenguaje configurado para razonamiento.
    """
    if GROQ_API_KEY and not GROQ_API_KEY.startswith("YOUR_"):
        return ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model=DEFAULT_GROQ_MODEL,
            temperature=temperature
        )
    elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("YOUR_"):
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=temperature
        )
    else:
        raise ValueError(
            "No se encontró GROQ_API_KEY ni OPENAI_API_KEY configurada en el archivo .env. "
            "Por favor proporciona al menos una de estas claves para el razonamiento del agente."
        )

