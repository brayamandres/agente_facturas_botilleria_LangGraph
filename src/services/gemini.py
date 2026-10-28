import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from colorist import BgColor
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.config import GOOGLE_API_KEY, DATA_OPTIMIZADO_DIR, DEFAULT_GEMINI_MODELS

# ============================================================================
# ESQUEMAS PYDANTIC DE EXTRACCIÓN (IE1, IE2 e IE5)
# ============================================================================

class ItemFacturaExtraido(BaseModel):
    """Esquema de validación para una línea de ítem extraída de la factura."""
    nombre_glosa: Optional[str] = Field(default=None, description="Glosa o descripción original del ítem")
    formato: Optional[str] = Field(default=None, description="Formato, calibre o empaque si existe (ej: 750cc, Pack x6)")
    cantidad: Optional[Any] = Field(default=None, description="Cantidad leída en la factura (debe validarse como entero estricto)")
    unidades: Optional[Any] = Field(default=None, description="Unidades por bulto/pack contenidas")
    es_pack: Optional[bool] = Field(default=None, description="Indica si es un bulto/pack o ítem unitario")
    precio_unitario: Optional[Any] = Field(default=None, description="Precio unitario o por pack listado en la factura")
    tipo_precio: Optional[str] = Field(default=None, description="POR_PACK | POR_UNIDAD | AMBIGUO")
    linea_id: Optional[int] = Field(default=None, description="Identificador correlativo de línea")


class FacturaExtraidaSchema(BaseModel):
    """Esquema de validación para el documento completo de factura."""
    proveedor: Optional[str] = Field(default=None, description="Nombre o razón social del proveedor")
    numero_factura: Optional[str] = Field(default=None, description="Folio o número de la factura")
    items: List[ItemFacturaExtraido] = Field(default_factory=list, description="Lista de ítems extraídos")
    tipo_documento: Optional[str] = Field(default="FACTURA", description="Tipo de documento detectado")
    es_documento_valido: bool = Field(default=True, description="Indica si corresponde a un documento formal")
    motivo_sin_items: Optional[str] = Field(default=None, description="Motivo si la factura no tiene ítems comerciales")
    estado_extraccion: Optional[str] = Field(default="OK", description="Estado de extracción: OK, DOC_SIN_ITEMS o ERROR_EXTRACCION")


def validar_esquema_extraccion(datos_raw: Any) -> Dict[str, Any]:
    """
    Valida y normaliza la extracción multimodal de una factura contra el esquema Pydantic
    (encabezado, lista de ítems, cantidades y precios).
    Distingue explícitamente entre:
    - 'OK': Documento válido con encabezado e ítems detectados.
    - 'DOC_SIN_ITEMS': Documento formal detectado (proveedor o folio legible), pero sin líneas comerciales.
    - 'ERROR_EXTRACCION': Fallo de lectura/OCR, JSON corrupto o imagen ilegible sin datos mínimos.
    """
    if not isinstance(datos_raw, dict):
        return {
            "proveedor": None,
            "numero_factura": None,
            "items": [],
            "tipo_documento": "DESCONOCIDO",
            "es_documento_valido": False,
            "motivo_sin_items": "La respuesta de extracción no es un diccionario válido.",
            "estado_extraccion": "ERROR_EXTRACCION",
        }

    if datos_raw.get("estado_extraccion") == "ERROR_EXTRACCION":
        return {
            "proveedor": datos_raw.get("proveedor"),
            "numero_factura": datos_raw.get("numero_factura"),
            "items": [],
            "tipo_documento": datos_raw.get("tipo_documento") or "ILEGIBLE",
            "es_documento_valido": False,
            "motivo_sin_items": datos_raw.get("motivo_sin_items") or "Extracción multimodal fallida.",
            "estado_extraccion": "ERROR_EXTRACCION",
        }

    try:
        esquema = FacturaExtraidaSchema(**datos_raw)
        res_dict = esquema.model_dump()
    except Exception as e:
        return {
            "proveedor": str(datos_raw.get("proveedor") or "") or None,
            "numero_factura": str(datos_raw.get("numero_factura") or "") or None,
            "items": [],
            "tipo_documento": "INVALIDO",
            "es_documento_valido": False,
            "motivo_sin_items": f"Error de validación contra esquema de factura: {e}",
            "estado_extraccion": "ERROR_EXTRACCION",
        }

    proveedor = res_dict.get("proveedor")
    folio = res_dict.get("numero_factura")
    items = res_dict.get("items") or []

    tiene_encabezado = bool(
        (proveedor and str(proveedor).strip().lower() not in ["none", "null", "s/f", "n/a", "desconocido", ""])
        or (folio and str(folio).strip().lower() not in ["none", "null", "s/f", "n/a", ""])
    )

    if not items:
        if tiene_encabezado:
            res_dict["estado_extraccion"] = "DOC_SIN_ITEMS"
            res_dict["motivo_sin_items"] = (
                res_dict.get("motivo_sin_items")
                or "Documento formal detectado con encabezado, pero sin ítems comerciales."
            )
        else:
            res_dict["estado_extraccion"] = "ERROR_EXTRACCION"
            res_dict["motivo_sin_items"] = "No se detectó encabezado legible ni ítems comerciales en el documento."
    else:
        res_dict["estado_extraccion"] = "OK"

    return res_dict

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
      "es_pack": false,
      "precio_unitario": 0,
      "tipo_precio": "POR_PACK | POR_UNIDAD | AMBIGUO | null"
    }
  ]
}

Restricciones:
- No inventes ni asumas datos ausentes (usa null).
- Conserva la glosa original del proveedor sin normalizar ni corregir ortografía.
- Devuelve únicamente el objeto JSON sin explicaciones adicionales.
- Si alguno de los items corresponde a costo de transporte, flete o despacho, ignóralo.
- En el campo 'unidades', indica el número exacto de unidades individuales contenidas por cada bulto/pack (ej: 6 si es pack de 6, 12 si es caja de 12; si es unitario suelto, 1; si es un bulto pero el número de unidades es ilegible o desconocido, usa null).
- En 'es_pack', indica true si el ítem corresponde a un pack, bulto, caja, sixpack, display o fardo; false si es unitario; null si es dudoso.
- En 'tipo_precio', indica 'POR_PACK' si el precio listado en la columna de la factura corresponde al bulto/pack completo, 'POR_UNIDAD' si ya viene expresado por cada botella/lata individual, o 'AMBIGUO' si la factura no lo aclara.
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
            datos_raw = json.loads(respuesta.text)
            return validar_esquema_extraccion(datos_raw)
        except Exception as e:
            print(f"{BgColor.WHITE}[Gemini Service]{BgColor.OFF} Aviso en modelo '{modelo}': {e}. Probando siguiente modelo...")

    return validar_esquema_extraccion({
        "proveedor": None,
        "numero_factura": None,
        "items": [],
        "tipo_documento": "ILEGIBLE",
        "es_documento_valido": False,
        "motivo_sin_items": "No fue posible extraer los datos de la factura con los modelos de Gemini configurados.",
        "estado_extraccion": "ERROR_EXTRACCION",
    })
