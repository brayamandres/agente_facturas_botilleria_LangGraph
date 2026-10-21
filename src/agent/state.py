"""
Definición del Estado del Agente (AgentState) para LangGraph.

El estado actúa como la memoria de trabajo compartida a lo largo de todos los nodos
del grafo, manteniendo trazabilidad completa de la factura, ítems pendientes,
decisiones de homologación y eventos de auditoría.
"""
from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    """
    Estado global del flujo de trabajo del agente de facturas.
    
    Args:
        TypedDict
        
    Returns:
        AgentState: Diccionario tipado que representa el estado del agente.
    """
    # 1. Datos de entrada

    ruta_imagen: str

    # 2. Percepción y extracción multimodal (Gemini)

    datos_factura: Dict[str, Any] #{'proveedor': str, 'numero_factura': str, 'items': list}.
    
    # 3. Control de duplicados (Condición adaptativa de negocio)

    es_duplicada: bool
    info_duplicada: Optional[Dict[str, Any]]

    # 4. Planificación y control de iteración de ítems

    items_pendientes: List[Dict[str, Any]]
    item_actual: Optional[Dict[str, Any]]           #Ítem que se encuentra en evaluación en el ciclo actual del grafo.
    candidatos_actuales: List[Dict[str, Any]]       #top 3 candidatos

    # 5. Razonamiento asistido por LLM y análisis de precios

    decision_actual: Optional[Dict[str, Any]]   #Resultado del LLM: {'sku_seleccionado', 'confianza', 'justificacion', 'requiere_revision'}.
    alerta_precio: Optional[Dict[str, Any]]     #Resultado del análisis de costo: {'alerta': bool, 'variacion_porcentual': float, 'mensaje': str}.
    

    # 6. Resultados acumulados y control de Human-in-the-Loop

    items_procesados: List[Dict[str, Any]]      #historial actualizados o creados en Firestore
    requiere_intervencion: bool
    """Bandera booleana que indica si el flujo requiere pausa para decisión humana."""

    datos_intervencion_humana: Optional[Dict[str, Any]]
    """Datos proporcionados por el usuario durante la revisión (ej. SKU nuevo, precio venta, confirmación)."""

    # 7. Memoria de largo plazo y auditoría
    auditoria_id: Optional[str]
    """ID del documento creado en la colección 'historial_facturas' de Firestore."""

    mensajes_bitacora: List[str]
    """Bitácora secuencial de pensamientos, acciones y decisiones tomadas por el agente."""

    estado_final: str
    """Estado final del proceso: 'EN_PROCESO', 'COMPLETADO', 'DUPLICADO_IGNORADO' o 'ERROR'."""

