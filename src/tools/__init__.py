"""
Herramientas (Tools) del Agente de Facturas agrupadas por función.
Consulta (query_tools), razonamiento (reasoning_tools) y escritura (write_tools).
"""
from src.tools.query_tools import (
    extraer_factura_desde_imagen,
    consultar_duplicidad_factura,
    buscar_candidatos_vectoriales,
    consultar_producto_por_sku,
    consultar_equivalencia_proveedor,
    validar_esquema_extraccion,
    FacturaExtraidaSchema,
    ItemFacturaExtraido,
)
from src.tools.reasoning_tools import (
    razonar_homologacion,
    evaluar_variacion_costo,
    detectar_indicio_pack,
    validar_unidades_item,
    resolver_costo_unitario_linea,
)
from src.tools.write_tools import (
    actualizar_stock_y_costo,
    registrar_nuevo_producto,
    registrar_auditoria_factura,
    guardar_equivalencia_proveedor,
    construir_id_operacion,
    construir_clave_equivalencia,
)

ALL_TOOLS = [
    # Consulta / Percepción multimodal y memoria contextual
    extraer_factura_desde_imagen,
    consultar_duplicidad_factura,
    buscar_candidatos_vectoriales,
    consultar_producto_por_sku,
    consultar_equivalencia_proveedor,
    # Razonamiento
    razonar_homologacion,
    evaluar_variacion_costo,
    # Escritura transaccional y aprendizaje de equivalencias
    actualizar_stock_y_costo,
    registrar_nuevo_producto,
    registrar_auditoria_factura,
    guardar_equivalencia_proveedor,
]
