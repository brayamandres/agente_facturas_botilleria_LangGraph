"""
Herramientas (Tools) del Agente de Facturas agrupadas por función.
Consulta (query_tools) , razonamiento (reasoning_tools) y escritura (write_tools)
"""
from src.tools.query_tools import (
    extraer_factura_desde_imagen,
    consultar_duplicidad_factura,
    buscar_candidatos_vectoriales,
    consultar_producto_por_sku,
)
from src.tools.reasoning_tools import (
    razonar_homologacion,
    evaluar_variacion_costo,
)
from src.tools.write_tools import (
    actualizar_stock_y_costo,
    registrar_nuevo_producto,
    registrar_auditoria_factura,
)

ALL_TOOLS = [
    # Consulta / Percepción multimodal
    extraer_factura_desde_imagen,
    consultar_duplicidad_factura,
    buscar_candidatos_vectoriales,
    consultar_producto_por_sku,
    # Razonamiento
    razonar_homologacion,
    evaluar_variacion_costo,
    # Escritura
    actualizar_stock_y_costo,
    registrar_nuevo_producto,
    registrar_auditoria_factura,
]

