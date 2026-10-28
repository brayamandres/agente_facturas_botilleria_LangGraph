"""
Módulo de Planificación Explícita y Política de Decisión Adaptativa (Criterio IE5).

Formaliza la política de prioridades operativas del agente según la pauta:
Prioridad | Condición | Acción
1 | Falla la verificación de duplicidad | Detener las escrituras.
2 | Faltan datos necesarios | Solicitar corrección.
3 | Hay equivalencia segura y datos válidos | Aplicar el movimiento.
4 | Hay ambigüedad o aumento de costo | Solicitar revisión.
5 | Falló un movimiento | Guardarlo como pendiente y reintentar sin repetir otros.
"""
from enum import IntEnum
from typing import Dict, Any, List, Optional
from colorist import BgColor, Color


class PrioridadPlanificacion(IntEnum):
    """Niveles de prioridad para la toma de decisiones adaptativas del agente."""
    P1_DUPLICIDAD_O_ERROR = 1
    P2_FALTAN_DATOS = 2
    P3_EQUIVALENCIA_SEGURA = 3
    P4_AMBIGUEDAD_O_COSTO = 4
    P5_FALLO_MOVIMIENTO = 5


TABLA_POLITICA_PLANIFICACION: List[Dict[str, Any]] = [
    {
        "prioridad": 1,
        "nombre": "Control de Duplicidad e Integridad de Proceso",
        "condicion": "Falla la verificación de duplicidad (factura ya completada o error en Firestore)",
        "accion": "Detener las escrituras y abortar el flujo para proteger el inventario.",
    },
    {
        "prioridad": 2,
        "nombre": "Integridad de Datos",
        "condicion": "Faltan datos necesarios (glosa vacía, cantidad no entera o precio ilegible)",
        "accion": "Solicitar corrección al operador humano (Human-in-the-Loop) sin asumir valores arbitrarios.",
    },
    {
        "prioridad": 3,
        "nombre": "Ejecución Segura",
        "condicion": "Hay equivalencia segura (memoria o confianza >= 0.75) y datos íntegros",
        "accion": "Aplicar el movimiento atómico (actualizar stock y movimiento) sin fricción.",
    },
    {
        "prioridad": 4,
        "nombre": "Gestión de Incertidumbre y Variación de Costo",
        "condicion": "Hay ambigüedad (confianza < 0.75, pack ambiguo) o aumento de costo (> 15%)",
        "accion": "Solicitar revisión y decisión humana con candidatos recuperados antes de escribir.",
    },
    {
        "prioridad": 5,
        "nombre": "Resiliencia e Idempotencia ante Fallos",
        "condicion": "Falló un movimiento transaccional durante el procesamiento",
        "accion": "Guardarlo como pendiente/fallido y permitir reintento selectivo sin repetir otros.",
    },
]


def formatear_aviso_planificacion(
    prioridad: int,
    regla: str,
    detalle: str,
) -> str:
    """
    Genera el mensaje estilizado con colorist para visibilizar la política de planificación explícita (IE5),
    destacando [Planificación] en color cyan y eliminando texto plano no diferenciado.
    """
    tag = f"{BgColor.CYAN}[Planificación]{BgColor.OFF}"
    pri_txt = f"{Color.CYAN}Prioridad {prioridad} ({regla}){Color.OFF}"
    return f"{tag} {pri_txt} -> {detalle}"


def evaluar_politica_evaluacion_item(
    item: Dict[str, Any],
    decision: Optional[Dict[str, Any]],
    alerta: Optional[Dict[str, Any]],
    campos_ilegibles: List[str],
) -> Dict[str, Any]:
    """
    Evalúa cuál regla de la política de planificación (P2, P3 o P4) gobierna la decisión
    adaptativa para la línea actual de la factura.
    """
    decision_dict = decision or {}
    alerta_dict = alerta or {}
    sku = decision_dict.get("sku_seleccionado")
    confianza = float(decision_dict.get("confianza") or 0.0)
    requiere_revision = bool(decision_dict.get("requiere_revision", False))
    hay_alerta_precio = bool(alerta_dict.get("alerta", False))
    ambiguedad_costo = bool(
        item.get("ambiguedad_costo")
        or alerta_dict.get("ambiguedad_pack")
        or item.get("unidades_ilegibles")
    )

    # PRIORIDAD 2: Faltan datos necesarios -> Solicitar corrección
    if len(campos_ilegibles) > 0:
        detalle = f"Faltan datos requeridos en el ítem: {', '.join(campos_ilegibles)}. Solicitando corrección al operador."
        return {
            "prioridad": PrioridadPlanificacion.P2_FALTAN_DATOS.value,
            "regla": "Integridad de Datos",
            "accion": "Solicitar corrección",
            "ruta_grafo": "intervencion_humana",
            "detalle": detalle,
            "mensaje_formateado": formatear_aviso_planificacion(2, "Integridad de Datos", detalle),
        }

    # PRIORIDAD 4: Hay ambigüedad o aumento de costo -> Solicitar revisión
    if not sku or requiere_revision or confianza < 0.75 or hay_alerta_precio or ambiguedad_costo:
        motivos = []
        if hay_alerta_precio:
            var_pct = alerta_dict.get("variacion_porcentual", 0.0)
            motivos.append(f"Aumento de costo detectado ({var_pct:+.1%})")
        if ambiguedad_costo:
            motivos.append("Ambigüedad en precio/unidades de pack")
        if not sku:
            motivos.append("No se encontró coincidencia en catálogo")
        elif confianza < 0.75 or requiere_revision:
            motivos.append(f"Coincidencia dudosa (Confianza: {confianza:.2f})")

        detalle = f"{'; '.join(motivos)}. Solicitando revisión humana."
        return {
            "prioridad": PrioridadPlanificacion.P4_AMBIGUEDAD_O_COSTO.value,
            "regla": "Gestión de Incertidumbre y Costos",
            "accion": "Solicitar revisión",
            "ruta_grafo": "intervencion_humana",
            "detalle": detalle,
            "mensaje_formateado": formatear_aviso_planificacion(4, "Gestión de Incertidumbre y Costos", detalle),
        }

    # PRIORIDAD 3: Hay equivalencia segura y datos válidos -> Aplicar el movimiento
    detalle = f"Equivalencia segura validada para '{item.get('nombre_glosa')}' -> SKU '{sku}' (Confianza: {confianza:.2f}) y datos íntegros. Aplicando movimiento atómico."
    return {
        "prioridad": PrioridadPlanificacion.P3_EQUIVALENCIA_SEGURA.value,
        "regla": "Ejecución Segura",
        "accion": "Aplicar el movimiento",
        "ruta_grafo": "aplicar_cambios",
        "detalle": detalle,
        "mensaje_formateado": formatear_aviso_planificacion(3, "Ejecución Segura", detalle),
    }

