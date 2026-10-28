"""
Herramientas de Razonamiento (Reasoning Tools) para el agente.
Permiten evaluar equivalencias comerciales, homologación con LLM y análisis de variación de precios.

Nota metodológica sobre la puntuación de confianza:
La métrica 'confianza' resultante combina heurísticamente la estimación reportada por el LLM (70%)
con la similitud vectorial coseno del candidato en Firestore (30%). Este valor representa una
puntuación heurística de confianza operativa para decidir umbrales de revisión humana (HITL),
y NO una probabilidad estadísticamente calibrada de acierto.
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
    confianza: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Puntuación heurística de confianza entre 0.0 y 1.0 (no es una probabilidad calibrada)",
    )
    justificacion: str = Field(default="", description="Motivo de la decisión")
    requiere_revision: bool = Field(default=True, description="Indica si requiere intervención humana")


class _DictHomologacionValidada(dict):
    """
    Diccionario estándar de 4 claves que conserva internamente metadatos de trazabilidad
    para evitar aplicar dos veces la ponderación heurística si se encadenan validaciones.
    """
    _heuristica_aplicada: bool = False
    _confianza_llm_original: float = 0.0


def calcular_puntuacion_heuristica_confianza(
    confianza_llm: float,
    similitud_vectorial: Optional[float] = None,
) -> float:
    """
    Calcula una única vez la puntuación heurística de confianza combinando:
    - 70% estimación del LLM sobre equivalencia comercial/abreviaciones.
    - 30% similitud vectorial coseno del candidato recuperado en Firestore.

    Importante: El valor devuelto en [0.0, 1.0] es un puntaje heurístico para priorizar
    revisión humana (umbral 0.75), no una probabilidad calibrada de acierto.
    """
    base_acotada = max(0.0, min(1.0, float(confianza_llm)))
    if isinstance(similitud_vectorial, (int, float)) and not isinstance(similitud_vectorial, bool):
        sim_float = float(similitud_vectorial)
        if 0.0 <= sim_float <= 1.0:
            return round((base_acotada * 0.7) + (sim_float * 0.3), 4)
    return round(base_acotada, 4)


def validar_estructura_homologacion(
    decision: Any,
    candidatos: Optional[List[Dict[str, Any]]] = None,
    umbral_confianza: float = 0.75,
) -> Dict[str, Any]:
    """
    Valida exclusivamente la estructura, tipos, rangos y pertenencia del SKU a los candidatos
    recuperados, SIN volver a modificar ni re-ponderar la puntuación de confianza.
    """
    if not isinstance(decision, dict):
        res_invalido = _DictHomologacionValidada(
            ResultadoHomologacion(
                sku_seleccionado=None,
                confianza=0.0,
                justificacion="Respuesta del modelo con formato inválido.",
                requiere_revision=True,
            ).model_dump()
        )
        res_invalido._heuristica_aplicada = True
        res_invalido._confianza_llm_original = 0.0
        return res_invalido

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

    if sku is not None and sku not in mapa_candidatos:
        justificacion = (
            f"SKU '{sku}' rechazado porque no pertenece a los candidatos recuperados "
            f"({list(mapa_candidatos.keys())}). {justificacion}"
        )
        sku = None
        confianza = 0.0
        requiere_revision = True

    # 6. Coherencia final de estado
    if sku is None:
        confianza = 0.0
        requiere_revision = True
    elif confianza < umbral_confianza:
        requiere_revision = True

    validado = ResultadoHomologacion(
        sku_seleccionado=sku,
        confianza=round(confianza, 4),
        justificacion=justificacion,
        requiere_revision=requiere_revision,
    )
    salida = _DictHomologacionValidada(validado.model_dump())
    salida._heuristica_aplicada = bool(getattr(decision, "_heuristica_aplicada", False))
    salida._confianza_llm_original = float(getattr(decision, "_confianza_llm_original", confianza))
    return salida


def validar_respuesta_homologacion(
    decision: Any,
    candidatos: Optional[List[Dict[str, Any]]] = None,
    umbral_confianza: float = 0.75,
) -> Dict[str, Any]:
    """
    Valida la estructura de la respuesta de homologación y aplica UNA SOLA VEZ el cálculo
    de la puntuación heurística de confianza (70% LLM + 30% similitud vectorial).
    """
    ya_ponderada = bool(isinstance(decision, dict) and getattr(decision, "_heuristica_aplicada", False))
    confianza_llm_base = (
        float(getattr(decision, "_confianza_llm_original", decision.get("confianza", 0.0) or 0.0))
        if isinstance(decision, dict)
        else 0.0
    )

    estructural = validar_estructura_homologacion(
        decision=decision,
        candidatos=candidatos,
        umbral_confianza=umbral_confianza,
    )

    sku = estructural.get("sku_seleccionado")
    if sku is not None and not ya_ponderada:
        lista_candidatos = candidatos if isinstance(candidatos, list) else []
        mapa_candidatos = {
            str(c.get("sku")).strip(): c
            for c in lista_candidatos
            if isinstance(c, dict) and c.get("sku")
        }
        candidato_match = mapa_candidatos.get(sku) or {}
        sim_vec = candidato_match.get("similitud_vectorial")

        puntuacion = calcular_puntuacion_heuristica_confianza(
            confianza_llm=estructural["confianza"],
            similitud_vectorial=sim_vec,
        )
        estructural["confianza"] = puntuacion
        estructural["requiere_revision"] = bool(
            estructural["requiere_revision"] or (puntuacion < umbral_confianza)
        )

    resultado_final = _DictHomologacionValidada(estructural)
    resultado_final._heuristica_aplicada = True
    resultado_final._confianza_llm_original = confianza_llm_base
    return resultado_final


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
2. Si uno de los candidatos coincide claramente en marca, tipo de licor y formato/calibre, entrega ÚNICAMENTE el "sku" exacto de ese candidato en "sku_seleccionado" y una puntuación heurística de "confianza" entre 0.0 y 1.0.
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
def razonar_homologacion(glosa_factura, formato_factura=None, candidatos=None):
    """
    Utiliza el LLM para razonar sobre abreviaciones y similitudes entre una glosa de factura y los candidatos recuperados.
    Valida estrictamente que el SKU seleccionado pertenezca a los candidatos y calcula una única vez la
    puntuación heurística de confianza (70% LLM + 30% similitud vectorial coseno).

    Args:
        glosa_factura (str): Glosa de la línea de factura a homologar.
        formato_factura (str | None): Formato detectado de la línea de factura (opcional).
        candidatos (list[dict] | None): Lista de candidatos del catálogo para comparar (opcional).

    Returns:
        dict: Diccionario validado con "sku_seleccionado", "confianza" (puntuación heurística),
              "justificacion" y "requiere_revision".
    """
    lista_candidatos = candidatos or []
    if not lista_candidatos:
        return validar_estructura_homologacion({
            "sku_seleccionado": None,
            "confianza": 0.0,
            "justificacion": "No hay candidatos disponibles en el catálogo para homologar.",
            "requiere_revision": True,
        }, candidatos=[])

    llm = get_reasoning_llm(temperature=0.0)
    cadena = PROMPT_HOMOLOGACION | llm | JsonOutputParser()

    resultado_crudo = cadena.invoke({
        "glosa_factura": glosa_factura,
        "formato_factura": formato_factura or "No especificado",
        "candidatos": json.dumps(lista_candidatos, indent=2, ensure_ascii=False),
    })

    return validar_respuesta_homologacion(resultado_crudo, candidatos=lista_candidatos)


import re

# ============================================================================
# NORMALIZACIÓN Y RESOLUCIÓN DE UNIDADES Y COSTO DE PACKS (IE1)
# ============================================================================

PATRON_PACK_TEXTO = re.compile(
    r"\b(pack|caja|display|bulto|sixpack|six-pack|fardo|bandeja|promo)\b",
    re.IGNORECASE,
)

PATRON_UNIDADES_EXPLICITAS = re.compile(
    r"(?:pack|caja|display|bulto|fardo|bandeja)\s*(?:x|\*|\:)?\s*(\d+)\b"
    r"|\b(\d+)\s*(?:pack|pk|bot|latas?|unid?)\b"
    r"|\b[xX]\s*(\d+)\b",
    re.IGNORECASE,
)


def detectar_indicio_pack(glosa: Any, formato: Any = None) -> tuple[bool, Optional[int]]:
    """
    Analiza la glosa y formato de una línea para detectar si corresponde a un bulto o pack,
    extrayendo el número de unidades si está explícito en el texto.

    Args:
        glosa (Any): Glosa o descripción de la línea de factura.
        formato (Any, optional): Formato o empaque detectado.

    Returns:
        tuple[bool, Optional[int]]: (es_pack, unidades_sugeridas_en_texto)
    """
    texto = f"{glosa or ''} {formato or ''}".strip()
    if not texto:
        return False, None

    unidades_sugeridas = None
    coincidencia = PATRON_UNIDADES_EXPLICITAS.search(texto)
    if coincidencia:
        for g in coincidencia.groups():
            if g and g.isdigit():
                num = int(g)
                if num > 1:
                    unidades_sugeridas = num
                    break

    es_pack = bool(PATRON_PACK_TEXTO.search(texto) or unidades_sugeridas is not None)
    return es_pack, unidades_sugeridas


def validar_unidades_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida el número de unidades individuales por pack o bulto.
    Para un producto que indique ser bulto o pack, NO asume 1 silenciosamente si el dato
    es ausente o inválido, marcándolo como ilegible para intervención humana.

    Args:
        item (dict): Diccionario de la línea de factura.

    Returns:
        dict: Diccionario con 'es_pack', 'unidades' (int o None), 'unidades_ilegibles' y 'motivo'.
    """
    glosa = str(item.get("nombre_glosa") or "").strip()
    formato = str(item.get("formato") or "").strip()

    es_pack_indicio, unidades_en_texto = detectar_indicio_pack(glosa, formato)
    es_pack_declarado = item.get("es_pack")

    unidades_raw = item.get("unidades")
    unidades_parseadas: Optional[int] = None
    if unidades_raw is not None:
        try:
            num = int(round(float(unidades_raw)))
            if num > 0:
                unidades_parseadas = num
        except (ValueError, TypeError):
            unidades_parseadas = None

    es_pack = bool(
        es_pack_declarado is True
        or es_pack_indicio
        or (unidades_parseadas is not None and unidades_parseadas > 1)
    )

    if not es_pack:
        return {
            "es_pack": False,
            "unidades": 1,
            "unidades_ilegibles": False,
            "origen_unidades": "UNITARIO_ESTANDAR",
            "motivo": "Producto unitario estándar.",
        }

    # Caso Pack:
    if unidades_parseadas is not None and unidades_parseadas > 1:
        return {
            "es_pack": True,
            "unidades": unidades_parseadas,
            "unidades_ilegibles": False,
            "origen_unidades": "VALOR_FACTURA",
            "motivo": f"Pack detectado con {unidades_parseadas} unidades por bulto.",
        }

    if unidades_en_texto is not None and unidades_en_texto > 1:
        return {
            "es_pack": True,
            "unidades": unidades_en_texto,
            "unidades_ilegibles": False,
            "origen_unidades": "TEXTO_GLOSA",
            "motivo": f"Pack detectado con {unidades_en_texto} unidades según glosa/formato.",
        }

    # Bulto detectado pero unidades ausentes o <= 1: ¡Ilegible!
    return {
        "es_pack": True,
        "unidades": None,
        "unidades_ilegibles": True,
        "origen_unidades": "ILEGIBLE",
        "motivo": "Bulto/pack detectado pero las unidades por pack son ausentes o inválidas.",
    }


def resolver_costo_unitario_linea(
    item: Dict[str, Any],
    sku: Optional[str] = None,
    costo_actual_catalogo: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Distingue explícitamente entre precio por pack y precio por unidad, y normaliza el costo
    unitario individual antes de compararlo o guardarlo en Firestore.
    Si la factura es ambigua, marca la decisión con 'ambiguo=True' para desviar a revisión humana.

    Args:
        item (dict): Línea de factura con precio_unitario, unidades, glosa, etc.
        sku (str | None, optional): SKU del catálogo (opcional).
        costo_actual_catalogo (float | None, optional): Costo unitario histórico en Firestore.

    Returns:
        dict: {
            "costo_unitario": float | None,
            "precio_factura_bruto": float | None,
            "precio_pack": float | None,
            "unidades": int | None,
            "es_pack": bool,
            "tipo_precio": "POR_PACK" | "POR_UNIDAD" | "AMBIGUO",
            "ambiguo": bool,
            "motivo": str,
        }
    """
    precio_raw = item.get("precio_unitario")
    precio_factura: Optional[float] = None
    if precio_raw is not None:
        try:
            val = float(precio_raw)
            if val > 0:
                precio_factura = val
        except (ValueError, TypeError):
            precio_factura = None

    info_unid = validar_unidades_item(item)
    es_pack = bool(info_unid.get("es_pack"))
    unidades = info_unid.get("unidades")
    tipo_declarado = str(item.get("tipo_precio") or "").strip().upper() or None

    if precio_factura is None:
        return {
            "costo_unitario": None,
            "precio_factura_bruto": None,
            "precio_pack": None,
            "unidades": unidades,
            "es_pack": es_pack,
            "tipo_precio": "AMBIGUO",
            "ambiguo": True,
            "motivo": "Precio unitario ilegible o ausente en la factura.",
        }

    if not es_pack or (unidades is not None and unidades <= 1):
        return {
            "costo_unitario": precio_factura,
            "precio_factura_bruto": precio_factura,
            "precio_pack": precio_factura,
            "unidades": 1,
            "es_pack": False,
            "tipo_precio": "POR_UNIDAD",
            "ambiguo": False,
            "motivo": "Precio unitario directo (ítem individual).",
        }

    if unidades is None or unidades <= 0:
        return {
            "costo_unitario": None,
            "precio_factura_bruto": precio_factura,
            "precio_pack": precio_factura,
            "unidades": None,
            "es_pack": True,
            "tipo_precio": "AMBIGUO",
            "ambiguo": True,
            "motivo": "No se puede calcular el costo unitario porque las unidades por pack son ilegibles.",
        }

    costo_si_pack = round(precio_factura / float(unidades), 2)
    costo_si_unidad = round(float(precio_factura), 2)

    # 1. Indicación explícita en factura / OCR
    if tipo_declarado in ["POR_PACK", "PACK"]:
        return {
            "costo_unitario": costo_si_pack,
            "precio_factura_bruto": precio_factura,
            "precio_pack": precio_factura,
            "unidades": unidades,
            "es_pack": True,
            "tipo_precio": "POR_PACK",
            "ambiguo": False,
            "motivo": f"Precio de factura por pack (${precio_factura:,.0f} / {unidades} = ${costo_si_pack:,.0f}/u).",
        }
    if tipo_declarado in ["POR_UNIDAD", "UNIDAD", "UNITARIO"]:
        return {
            "costo_unitario": costo_si_unidad,
            "precio_factura_bruto": precio_factura,
            "precio_pack": round(costo_si_unidad * unidades, 2),
            "unidades": unidades,
            "es_pack": True,
            "tipo_precio": "POR_UNIDAD",
            "ambiguo": False,
            "motivo": f"Precio de factura por unidad individual (${costo_si_unidad:,.0f}/u).",
        }

    # 2. Contraste con el costo histórico del catálogo
    costo_hist: float = 0.0
    if costo_actual_catalogo is not None:
        try:
            ch = float(costo_actual_catalogo)
            if ch > 0:
                costo_hist = ch
        except (ValueError, TypeError):
            costo_hist = 0.0

    if costo_hist > 0:
        err_pack = abs(costo_si_pack - costo_hist) / costo_hist
        err_unit = abs(costo_si_unidad - costo_hist) / costo_hist

        if err_pack <= 0.30 and err_unit >= 0.60:
            return {
                "costo_unitario": costo_si_pack,
                "precio_factura_bruto": precio_factura,
                "precio_pack": precio_factura,
                "unidades": unidades,
                "es_pack": True,
                "tipo_precio": "POR_PACK",
                "ambiguo": False,
                "motivo": (
                    f"Precio de factura (${precio_factura:,.0f}) identificado como precio por PACK de {unidades} unidades. "
                    f"Costo unitario normalizado: ${costo_si_pack:,.0f}/u (coincide con histórico ${costo_hist:,.0f})."
                ),
            }

        if err_unit <= 0.30 and err_pack >= 0.60:
            return {
                "costo_unitario": costo_si_unidad,
                "precio_factura_bruto": precio_factura,
                "precio_pack": round(costo_si_unidad * unidades, 2),
                "unidades": unidades,
                "es_pack": True,
                "tipo_precio": "POR_UNIDAD",
                "ambiguo": False,
                "motivo": (
                    f"Precio de factura (${precio_factura:,.0f}) ya viene expresado por unidad individual "
                    f"(coincide con histórico ${costo_hist:,.0f})."
                ),
            }

    # 3. Caso ambiguo (sin histórico concluyente o producto nuevo con pack)
    return {
        "costo_unitario": costo_si_pack,
        "precio_factura_bruto": precio_factura,
        "precio_pack": precio_factura,
        "unidades": unidades,
        "es_pack": True,
        "tipo_precio": "AMBIGUO",
        "ambiguo": True,
        "motivo": (
            f"Ambigüedad de facturación en pack de {unidades} unidades: "
            f"se requiere confirmar si ${precio_factura:,.0f} es por el pack completo (${costo_si_pack:,.0f}/u) "
            f"o por unidad individual (${costo_si_unidad:,.0f}/u)."
        ),
    }


@tool
def evaluar_variacion_costo(costo_factura, costo_actual, umbral_alerta=0.15):
    """Evalúa la variación porcentual entre el costo nuevo de la factura y el costo actual del catálogo.
    Si la variación supera el umbral (por defecto 15%), genera una alerta para revisar el precio de venta.

    Args:
        costo_factura (float): Costo unitario normalizado de la línea de factura.
        costo_actual (float): Costo unitario actual registrado en el catálogo.
        umbral_alerta (float, optional): Porcentaje de variación que dispara la alerta. Default con 0.15.

    Returns:
        dict: Diccionario con la alerta, variación porcentual, diferencia en pesos y mensaje de recomendación.
    """
    if costo_actual <= 0:
        return {
            "alerta": False,
            "variacion_porcentual": 0.0,
            "mensaje": "No hay costo anterior registrado para comparar.",
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
        "mensaje": mensaje,
    }
