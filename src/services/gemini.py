import json
from pathlib import Path
from PIL import Image
from colorist import BgColor
from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, DATA_OPTIMIZADO_DIR, DEFAULT_GEMINI_MODELS

PROMPT_EXTRACCION = """Rol: Eres un asistente experto en lectura y extracción de datos de facturas de distribuidores de licores en Chile.

Tarea: Extrae el encabezado y el detalle de ítems comprados en la factura de la imagen.

Formato de salida (JSON estricto):
{
  "proveedor": "Nombre del proveedor o null",
  "numero_factura": "Número o folio de la factura o null",
  "items": [
    {
      "nombre_glosa": "Glosa o descripción original del ítem",
      "formato": "Formato, calibre o empaque si existe (ej: 750cc, Pack x6) o null",
      "cantidad": 0,
      "unidades": 0,
      "precio_unitario": 0
    }
  ]
}

Restricciones:
- No inventes ni asumas datos ausentes (usa null).
- Conserva la glosa original del proveedor sin normalizar ni corregir ortografía.
- Devuelve únicamente el objeto JSON sin explicaciones adicionales.
- Si alguno de los items corresponde a costo de transporte, flete o despacho, ignóralo.
- En el campo 'unidades', indica cuántas unidades vienen en caso de que se trate de un pack o bulto (si es unitario, 1).
"""

_gemini_client = None

def get_gemini_client():
    """Retorna la instancia del cliente Google Gemini."""
    global _gemini_client
    if _gemini_client is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY no está configurada en el archivo .env.")
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client


def preprocesar_imagen_factura(ruta_imagen, max_size=(1280, 720)):
    """
    Optimiza la imagen de la factura redimensionándola manteniendo el aspect ratio
    para reducir consumo de ancho de banda y latencia sin perder legibilidad OCR.
    
    Args:
        ruta_imagen (str | Path): Ruta local o URL de la imagen de la factura.
        max_size (tuple[int, int], optional): Tamaño máximo (ancho, alto) para redimensionar. Default es (1280, 720).
    Returns:
        Path: Ruta de la imagen optimizada.
    """
    ruta_original = Path(ruta_imagen)
    if not ruta_original.exists():
        raise FileNotFoundError(f"No se encontró la imagen de factura en: {ruta_original}")

    img = Image.open(ruta_original)
    img.thumbnail(max_size)

    ruta_salida = DATA_OPTIMIZADO_DIR / f"opt_{ruta_original.name}"
    img.save(ruta_salida)
    return ruta_salida


def extraer_datos_factura(ruta_imagen):
    """
    Lee visualmente la factura con Gemini y devuelve los datos estructurados en un diccionario.
    
    Args:
        ruta_imagen (str | Path): Ruta local o URL de la imagen de la factura a procesar.
    Returns:
        dict: Diccionario con los datos extraídos de la factura, incluyendo proveedor, número de factura y lista de ítems.
    """
    ruta_optimizada = preprocesar_imagen_factura(ruta_imagen)
    imagen = Image.open(ruta_optimizada)
    client = get_gemini_client()

    for modelo in DEFAULT_GEMINI_MODELS:
        try:
            respuesta = client.models.generate_content(
                model=modelo,
                contents=[PROMPT_EXTRACCION, imagen],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )
            return json.loads(respuesta.text)
        except Exception as e:
            print(f"{BgColor.WHITE}[Gemini Service]{BgColor.OFF} Aviso en modelo '{modelo}': {e}. Probando siguiente modelo...")

    raise RuntimeError("No fue posible extraer los datos de la factura con los modelos de Gemini configurados.")
