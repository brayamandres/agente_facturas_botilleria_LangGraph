"""
Herramientas de Razonamiento (Reasoning Tools) para el agente.
Permiten evaluar equivalencias comerciales, homologación con LLM y análisis de variación de precios.
"""
import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.services.llm import get_reasoning_llm

#PROMPT

#########################################################################################
PROMPT_HOMOLOGACION = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente experto en inventarios de botillería y distribución de licores en Chile. Responde exclusivamente en formato JSON."),
    ("human", """Tu tarea es homologar una línea cruda de factura de proveedor con uno de los productos de nuestro catálogo interno.

LÍNEA DE FACTURA:
- Glosa original: "{glosa_factura}"
- Formato detectado: "{formato_factura}"

CANDIDATOS DEL CATÁLOGO RECUPERADOS:
{candidatos}

INSTRUCCIONES:
1. Evalúa abreviaciones comunes de distribuidores chilenos (ej: "mst" = Mistral, "cerv cor" = Cerveza Corona, "7lts" o "750" = 750cc, "35" = 35°).
2. Si uno de los candidatos coincide claramente en marca, tipo de licor y formato/calibre, entrega su "sku_seleccionado" y un nivel de "confianza" entre 0.0 y 1.0.
3. Si ninguno coincide o la descripción es muy ambigua/incompleta, define "sku_seleccionado" como null y marca "requiere_revision" como true.

Responde ÚNICAMENTE en formato JSON con la siguiente estructura:
{{
  "sku_seleccionado": "SKU o null",
  "confianza": 0.95,
  "justificacion": "Breve razón del match",
  "requiere_revision": false
}}
""")
])
#########################################################################################

#TOOLS

@tool
def razonar_homologacion(glosa_factura: str, formato_factura: str, candidatos: list[dict]) -> dict:
    """
    Utiliza el LLM para razonar sobre abreviaciones y similitudes entre una glosa de factura y los candidatos recuperados.
    Retorna el SKU seleccionado, nivel de confianza (0.0 a 1.0), justificación y si requiere revisión humana.
    """
    llm = get_reasoning_llm(temperature=0.0)
    cadena = PROMPT_HOMOLOGACION | llm | JsonOutputParser()

    resultado = cadena.invoke({
        "glosa_factura": glosa_factura,
        "formato_factura": formato_factura or "No especificado",
        "candidatos": json.dumps(candidatos, indent=2, ensure_ascii=False)
    })
    return resultado


@tool
def evaluar_variacion_costo(costo_factura: float, costo_actual: float, umbral_alerta: float = 0.15) -> dict:
    """
    Evalúa la variación porcentual entre el costo nuevo de la factura y el costo actual del catálogo.
    Si la variación supera el umbral (por defecto 15%), genera una alerta para revisar el precio de venta.
    """
    if costo_actual <= 0:
        return {
            "alerta": False,
            "variacion_porcentual": 0.0,
            "mensaje": "No hay costo anterior registrado para comparar."
        }

    diferencia = costo_factura - costo_actual
    variacion = diferencia / costo_actual

    alerta = variacion > umbral_alerta
    mensaje = (
        f"Alerta de variación de costo: Subió un {variacion * 100:.1f}% "
        f"(de ${costo_actual:,.0f} a ${costo_factura:,.0f}). Se sugiere ajustar el precio de venta."
        if alerta else
        f"Variación de costo dentro del margen normal ({variacion * 100:.1f}%)."
    )

    return {
        "alerta": alerta,
        "variacion_porcentual": round(variacion * 100, 2),
        "diferencia_pesos": diferencia,
        "mensaje": mensaje
    }

