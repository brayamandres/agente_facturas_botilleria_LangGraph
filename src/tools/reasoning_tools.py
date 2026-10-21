"""
Herramientas de Razonamiento (Reasoning Tools) para el agente.
Permiten evaluar equivalencias comerciales, homologación con LLM y análisis de variación de precios.
"""
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.services.llm import get_reasoning_llm


class ResultadoHomologacion(BaseModel):
    """Esquema estructurado y validado para la decisión de homologación."""
    sku_seleccionado: Optional[str] = Field(default=None, description="SKU del candidato seleccionado o None")
    confianza: float = Field(default=0.0, ge=0.0, le=1.0, description="Nivel de confianza entre 0.0 y 1.0")
    justificacion: str = Field(default="", description="Motivo de la decisión")
    requiere_revision: bool = Field(default=True, description="Indica si requiere intervención humana")


def validar_respuesta_homologacion(
    decision: Any,
    candidatos: Optional[List[Dict[str, Any]]] = None,
    umbral_confianza: float = 0.75
) -> Dict[str, Any]:
    """
    Valida estrictamente que la respuesta de homologación cumpla tipos, rangos
    y que el SKU seleccionado pertenezca efectivamente a la lista de candidatos recuperados.
    Si el candidato incluye 'similitud_vectorial', calibra la confianza del LLM con la similitud coseno.
    """
    if not isinstance(decision, dict):
        return {
            "sku_seleccionado": None,
            "confianza": 0.0,
            "justificacion": "Respuesta del modelo con formato inválido.",
            "requiere_revision": True
        }

    # 1. Normalizar SKU
    sku_raw = decision.get("sku_seleccionado")
    sku: Optional[str] = None
    if isinstance(sku_raw, str):
        sku_limpio = sku_raw.strip()
        if sku_limpio and sku_limpio.lower() not in ["null", "none", "n/a", "ninguno"]:
            sku = sku_limpio

    # 2. Normalizar justificación
    just_raw = decision.get("justificacion")
    justificacion = str(just_raw).strip() if just_raw is not None else ""
    if not justificacion:
        justificacion = "Sin justificación proporcionada."

    # 3. Normalizar y acotar confianza en [0.0, 1.0]
    conf_raw = decision.get("confianza", 0.0)
    try:
        confianza = float(conf_raw) if conf_raw is not None and not isinstance(conf_raw, bool) else 0.0
    except (ValueError, TypeError):
        confianza = 0.0
    confianza = max(0.0, min(1.0, confianza))

    # 4. Normalizar requiere_revision
    rev_raw = decision.get("requiere_revision", False)
    if isinstance(rev_raw, bool):
        requiere_revision = rev_raw
    elif isinstance(rev_raw, str):
        requiere_revision = rev_raw.strip().lower() in ["true", "1", "si", "sí", "yes"]
    else:
        requiere_revision = bool(rev_raw)

    # 5. Validar pertenencia del SKU a los candidatos recuperados
    lista_candidatos = candidatos if isinstance(candidatos, list) else []
    mapa_candidatos = {
        str(c.get("sku")).strip(): c
        for c in lista_candidatos
        if isinstance(c, dict) and c.get("sku")
    }

    if sku is not None:
        if sku not in mapa_candidatos:
            justificacion = (
                f"SKU '{sku}' rechazado porque no pertenece a los candidatos recuperados "
                f"({list(mapa_candidatos.keys())}). {justificacion}"
            )
            sku = None
            confianza = 0.0
            requiere_revision = True
        else:
            # 6. Calibrar confianza si el candidato aporta similitud vectorial objetiva
            candidato_match = mapa_candidatos[sku]
            sim_vec = candidato_match.get("similitud_vectorial")
            if isinstance(sim_vec, (int, float)) and 0.0 <= float(sim_vec) <= 1.0:
                confianza = round((confianza * 0.7) + (float(sim_vec) * 0.3), 4)

    # 7. Coherencia final de estado
    if sku is None:
        confianza = 0.0
        requiere_revision = True
    elif confianza < umbral_confianza:
        requiere_revision = True

    validado = ResultadoHomologacion(
        sku_seleccionado=sku,
        confianza=round(confianza, 4),
        justificacion=justificacion,
        requiere_revision=requiere_revision
    )
    return validado.model_dump()


# PROMPT
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
2. Si uno de los candidatos coincide claramente en marca, tipo de licor y formato/calibre, entrega ÚNICAMENTE el "sku" exacto de ese candidato en "sku_seleccionado" y un nivel de "confianza" entre 0.0 y 1.0.
3. No inventes SKUs que no estén en la lista de CANDIDATOS DEL CATÁLOGO RECUPERADOS.
4. Si ninguno coincide o la descripción es muy ambigua/incompleta, define "sku_seleccionado" como null y marca "requiere_revision" como true.

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


@tool
def razonar_homologacion(glosa_factura, formato_factura = None, candidatos = None):
    """
    Utiliza el LLM para razonar sobre abreviaciones y similitudes entre una glosa de factura y los candidatos recuperados.
    Valida estrictamente que el SKU seleccionado pertenezca a los candidatos y que los tipos/rangos sean válidos.
    
    Args:
        glosa_factura (str): Glosa de la línea de factura a homologar.
        formato_factura (str | None): Formato detectado de la línea de factura (opcional).
        candidatos (list[dict] | None): Lista de candidatos del catálogo para comparar (opcional).
        
    Returns:
        dict: Diccionario validado con "sku_seleccionado", "confianza", "justificacion" y "requiere_revision".
    """
    lista_candidatos = candidatos or []
    if not lista_candidatos:
        return validar_respuesta_homologacion({
            "sku_seleccionado": None,
            "confianza": 0.0,
            "justificacion": "No hay candidatos disponibles en el catálogo para homologar.",
            "requiere_revision": True
        }, candidatos=[])

    llm = get_reasoning_llm(temperature=0.0)
    cadena = PROMPT_HOMOLOGACION | llm | JsonOutputParser()

    resultado_crudo = cadena.invoke({
        "glosa_factura": glosa_factura,
        "formato_factura": formato_factura or "No especificado",
        "candidatos": json.dumps(lista_candidatos, indent=2, ensure_ascii=False)
    })

    return validar_respuesta_homologacion(resultado_crudo, candidatos=lista_candidatos)


@tool
def evaluar_variacion_costo(costo_factura, costo_actual, umbral_alerta = 0.15): 
    """Evalúa la variación porcentual entre el costo nuevo de la factura y el costo actual del catálogo.
    Si la variación supera el umbral (por defecto 15%), genera una alerta para revisar el precio de venta.

    Args:
        costo_factura (float): Costo unitario de la línea de factura.
        costo_actual (float): Costo unitario actual registrado en el catálogo.
        umbral_alerta (float, optional): Porcentaje de variación que dispara la alerta. Default con 0.15.
        
    Returns:
        dict: Diccionario con la alerta, variación porcentual, diferencia en pesos y mensaje de recomendación.
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
