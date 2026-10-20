"""
Orquestador de Flujo del Agente de Facturas con LangGraph.

Implementa una Máquina de Estados Finita (StateGraph) con:
- Nodos especializados que ejecutan las 9 herramientas creadas.
- Bordes condicionales para toma de decisiones adaptativas (duplicados, anomalías de precios, baja confianza).
- Ciclos de iteración (loops) para procesar múltiples ítems secuencialmente.
- Memoria de corto plazo (MemorySaver checkpointer) para dar soporte a Human-in-the-Loop.
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.tools import (
    extraer_factura_desde_imagen,
    consultar_duplicidad_factura,
    buscar_candidatos_vectoriales,
    consultar_producto_por_sku,
    razonar_homologacion,
    evaluar_variacion_costo,
    actualizar_stock_y_costo,
    registrar_nuevo_producto,
    registrar_auditoria_factura,
)


# ============================================================================
# 1. NODOS DEL GRAFO (Execution Nodes)
# ============================================================================

def node_extraer(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 1: Invoca la herramienta multimodal para extraer encabezado e ítems desde la imagen.
    """
    ruta_imagen = state["ruta_imagen"]
    print(f"\n[Agente - Percepción] Extrayendo datos de la factura: '{ruta_imagen}'...")

    datos_factura = extraer_factura_desde_imagen.invoke({"ruta_imagen": ruta_imagen})
    items = list(datos_factura.get("items", []))

    msg = f"Factura extraída con éxito. Proveedor: '{datos_factura.get('proveedor')}', Folio: '{datos_factura.get('numero_factura')}', Ítems detectados: {len(items)}"
    print(f" -> {msg}")

    return {
        "datos_factura": datos_factura,
        "items_pendientes": items,
        "items_procesados": [],
        "mensajes_bitacora": state.get("mensajes_bitacora", []) + [msg],
        "estado_final": "EN_PROCESO"
    }


def node_verificar_duplicidad(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 2: Verifica en la memoria de largo plazo (Firestore) si la factura ya fue procesada.
    """
    datos = state["datos_factura"]
    folio = datos.get("numero_factura")
    proveedor = datos.get("proveedor")

    print(f"[Agente - Verificación] Comprobando duplicidad para folio '{folio}' de '{proveedor}'...")
    resultado = consultar_duplicidad_factura.invoke({"folio": folio, "proveedor": proveedor})

    es_duplicada = resultado.get("es_duplicada", False)
    msg = resultado.get("mensaje", "")
    print(f" -> {msg}")

    return {
        "es_duplicada": es_duplicada,
        "info_duplicada": resultado.get("registro_existente"),
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_manejar_duplicada(state: AgentState) -> Dict[str, Any]:
    """
    Nodo Alternativo: Detiene el flujo adaptativamente si la factura ya existe,
    evitando la doble contabilización en inventario.
    """
    info = state.get("info_duplicada") or {}
    msg = f"ADVERTENCIA: Proceso abortado para evitar duplicar stock. Factura folio '{state['datos_factura'].get('numero_factura')}' registrada el {info.get('fecha')}."
    print(f"\n[Agente - Control] {msg}")

    return {
        "estado_final": "DUPLICADO_IGNORADO",
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_preparar_siguiente_item(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 3 (Loop): Toma el primer ítem de la lista de pendientes para su evaluación individual.
    """
    pendientes = state["items_pendientes"]
    item_actual = pendientes[0]
    restantes = pendientes[1:]

    msg = f"Iniciando evaluación de ítem: '{item_actual.get('nombre_glosa')}' (Formato: {item_actual.get('formato')}, Cantidad: {item_actual.get('cantidad')})"
    print(f"\n--- {msg} ---")

    return {
        "item_actual": item_actual,
        "items_pendientes": restantes,
        "candidatos_actuales": [],
        "decision_actual": None,
        "alerta_precio": None,
        "requiere_intervencion": False,
        "datos_intervencion_humana": None,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_recuperar_candidatos(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 4: Recuperación semántica de los top 3 candidatos más similares en Firestore.
    """
    item = state["item_actual"]
    glosa = item.get("nombre_glosa", "")
    formato = item.get("formato", "")

    print(f"[Agente - Retrieval Semántico] Buscando candidatos para '{glosa}'...")
    candidatos = buscar_candidatos_vectoriales.invoke({"glosa": glosa, "formato": formato, "top_k": 3})

    nombres_candidatos = [f"{c['sku']} ({c['nombre']})" for c in candidatos]
    msg = f"Candidatos recuperados: {', '.join(nombres_candidatos) if nombres_candidatos else 'Ninguno'}"
    print(f" -> {msg}")

    return {
        "candidatos_actuales": candidatos,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_razonar_homologacion(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 5: El LLM razona sobre abreviaciones y similitud con los candidatos recuperados.
    """
    item = state["item_actual"]
    candidatos = state["candidatos_actuales"]

    print("[Agente - Razonamiento LLM] Evaluando homologación...")
    decision = razonar_homologacion.invoke({
        "glosa_factura": item.get("nombre_glosa", ""),
        "formato_factura": item.get("formato", ""),
        "candidatos": candidatos
    })

    sku = decision.get("sku_seleccionado")
    confianza = decision.get("confianza", 0.0)
    justificacion = decision.get("justificacion", "")

    msg = f"Decisión de homologación: SKU={sku} (Confianza={confianza:.2f}) | Motivo: {justificacion}"
    print(f" -> {msg}")

    return {
        "decision_actual": decision,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_evaluar_costo(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 6: Evalúa si existe variación abrupta de costo en comparación al histórico.
    """
    decision = state["decision_actual"] or {}
    item = state["item_actual"]
    sku = decision.get("sku_seleccionado")

    alerta_precio = None
    if sku:
        prod_info = consultar_producto_por_sku.invoke({"sku": sku})
        if prod_info.get("encontrado"):
            costo_actual = prod_info["producto"].get("precio_costo", 0)
            costo_factura = float(item.get("precio_unitario", 0))

            alerta_precio = evaluar_variacion_costo.invoke({
                "costo_factura": costo_factura,
                "costo_actual": costo_actual,
                "umbral_alerta": 0.15
            })

            if alerta_precio.get("alerta"):
                print(f" -> [ALERTA DE COSTO] {alerta_precio.get('mensaje')}")

    return {
        "alerta_precio": alerta_precio
    }


def node_intervencion_humana(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 7: Human-in-the-Loop.
    Se activa cuando:
    - La confianza es baja (<0.75) o el SKU es None (producto no existe).
    - El LLM marcó 'requiere_revision: true'.
    - Hubo una alerta crítica de costo (>15%).
    """
    item = state["item_actual"]
    decision = state["decision_actual"] or {}
    alerta = state["alerta_precio"] or {}
    sku = decision.get("sku_seleccionado")
    glosa = item.get("nombre_glosa", "")
    formato = item.get("formato", "")
    costo_factura = float(item.get("precio_unitario", 0))
    cantidad = int(item.get("cantidad", 1))

    print("\n" + "="*70)
    print(" [HUMAN-IN-THE-LOOP] Se requiere intervención humana para este ítem:")
    print(f" Glosa Factura: '{glosa}' | Formato: {formato} | Costo Factura: ${costo_factura} | Cantidad: {cantidad}")

    # CASO 1: Producto no encontrado o no homologable -> Dar de alta nuevo producto
    if not sku or decision.get("requiere_revision") or decision.get("confianza", 0) < 0.75:
        print(f" Motivo: No hay coincidencia segura en el catálogo (Confianza: {decision.get('confianza', 0.0):.2f}).")
        print(f" Justificación: {decision.get('justificacion')}")

        resp = input(" -> ¿Deseas agregar este producto como NUEVO en el catálogo? (s/n): ").strip().lower()

        if resp in ["s", "si", "y", "yes"]:
            sku_sug = glosa.upper().replace(" ", "-")[:15]
            nuevo_sku = input(f"    * SKU único [Enter para '{sku_sug}']: ").strip().upper() or sku_sug
            nombre = input(f"    * Nombre comercial [Enter para '{glosa}']: ").strip() or glosa
            marca = input("    * Marca: ").strip() or "Genérico"
            categoria = input("    * Categoría (Destilados, Vinos, Cervezas, Abarrotes): ").strip() or "Otros"
            subcategoria = input("    * Subcategoría: ").strip() or "General"
            nuevo_formato = input(f"    * Formato [Enter para '{formato or 'Unidad'}']: ").strip() or (formato or "Unidad")

            while True:
                pv_input = input(f"    * Precio de venta al público (Costo factura: ${costo_factura}): ").strip()
                try:
                    precio_venta = float(pv_input)
                    break
                except ValueError:
                    print("      Ingresa un número válido.")

            datos_accion = {
                "tipo_accion": "CREAR_NUEVO",
                "sku": nuevo_sku,
                "nombre": nombre,
                "marca": marca,
                "categoria": categoria,
                "subcategoria": subcategoria,
                "formato": nuevo_formato,
                "precio_costo": costo_factura,
                "precio_venta": precio_venta,
                "stock_inicial": cantidad
            }
        else:
            print(" -> Ítem omitido por decisión del operador.")
            datos_accion = {"tipo_accion": "OMITIR", "sku": None}

    # CASO 2: Producto homologado pero con alerta de precio
    else:
        print(f" Motivo: Alerta de variación de costo detectada ({alerta.get('variacion_porcentual')}%)")
        print(f" Producto: {sku}")
        pv_mod = input(" -> ¿Deseas ajustar el precio de venta? [Escribe nuevo precio o Enter para mantener]: ").strip()
        nuevo_pv = float(pv_mod) if pv_mod and pv_mod.replace('.', '', 1).isdigit() else None

        datos_accion = {
            "tipo_accion": "ACTUALIZAR_EXISTENTE",
            "sku": sku,
            "cantidad": cantidad,
            "precio_costo": costo_factura,
            "nuevo_precio_venta": nuevo_pv
        }

    print("="*70 + "\n")

    return {
        "datos_intervencion_humana": datos_accion,
        "requiere_intervencion": False
    }


def node_aplicar_cambios(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 8: Aplica las escrituras atómicas en Firestore (actualizar stock o crear nuevo producto).
    """
    item = state["item_actual"]
    decision = state["decision_actual"] or {}
    accion = state.get("datos_intervencion_humana")

    registro_item = {
        "glosa_original": item.get("nombre_glosa"),
        "formato": item.get("formato"),
        "cantidad": item.get("cantidad", 0),
        "precio_unitario": item.get("precio_unitario", 0),
        "sku_homologado": None,
        "estado": "PENDIENTE"
    }

    # Si vino de intervención humana
    if accion:
        tipo = accion.get("tipo_accion")
        if tipo == "CREAR_NUEVO":
            resultado = registrar_nuevo_producto.invoke({
                "sku": accion["sku"],
                "nombre": accion["nombre"],
                "marca": accion["marca"],
                "categoria": accion["categoria"],
                "subcategoria": accion["subcategoria"],
                "formato": accion["formato"],
                "precio_costo": accion["precio_costo"],
                "precio_venta": accion["precio_venta"],
                "stock_inicial": accion["stock_inicial"],
            })
            registro_item["sku_homologado"] = accion["sku"]
            registro_item["estado"] = "CREADO_NUEVO"
            print(f"[Agente - Escritura] {resultado.get('mensaje')}")

        elif tipo == "ACTUALIZAR_EXISTENTE":
            resultado = actualizar_stock_y_costo.invoke({
                "sku": accion["sku"],
                "cantidad": accion["cantidad"],
                "nuevo_costo": accion["precio_costo"],
                "nuevo_precio_venta": accion.get("nuevo_precio_venta")
            })
            registro_item["sku_homologado"] = accion["sku"]
            registro_item["estado"] = "ACTUALIZADO"
            print(f"[Agente - Escritura] {resultado.get('mensaje')}")

        else:
            registro_item["estado"] = "OMITIDO_POR_HUMANO"
            print("[Agente - Escritura] Ítem marcado como omitido.")

    # Si fue homologado automáticamente sin intervención
    else:
        sku = decision.get("sku_seleccionado")
        cantidad = int(item.get("cantidad", 1))
        costo = float(item.get("precio_unitario", 0))

        resultado = actualizar_stock_y_costo.invoke({
            "sku": sku,
            "cantidad": cantidad,
            "nuevo_costo": costo
        })
        registro_item["sku_homologado"] = sku
        registro_item["estado"] = "AUTO_HOMOLOGADO"
        print(f"[Agente - Escritura] {resultado.get('mensaje')}")

    items_actualizados = list(state.get("items_procesados", [])) + [registro_item]
    return {
        "items_procesados": items_actualizados
    }


def node_registrar_auditoria(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 9: Finaliza el flujo guardando el comprobante de auditoría en la colección
    'historial_facturas' de Firestore (Memoria transaccional de largo plazo).
    """
    datos = state["datos_factura"]
    folio = datos.get("numero_factura", "S/F")
    proveedor = datos.get("proveedor", "DESCONOCIDO")
    items_procesados = state.get("items_procesados", [])

    print(f"\n[Agente - Auditoría] Guardando registro transaccional para folio '{folio}'...")

    resultado = registrar_auditoria_factura.invoke({
        "folio": folio,
        "proveedor": proveedor,
        "total_items": len(items_procesados),
        "items_procesados": items_procesados,
        "estado": "COMPLETADO"
    })

    id_auditoria = resultado.get("id_auditoria")
    msg = f"Factura folio '{folio}' procesada completamente. Auditoría ID: {id_auditoria}"
    print(f" -> {msg}\n")

    return {
        "auditoria_id": id_auditoria,
        "estado_final": "COMPLETADO",
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


# ============================================================================
# 2. BORDES CONDICIONALES (Adaptive Decision Edges)
# ============================================================================

def decidir_ruta_duplicidad(state: AgentState) -> str:
    """
    Borde condicional 1: Si la factura ya existe, salta al manejador de duplicados.
    Si es nueva, arranca la iteración de productos.
    """
    if state.get("es_duplicada"):
        return "manejar_duplicada"
    return "preparar_siguiente_item"


def decidir_ruta_evaluacion(state: AgentState) -> str:
    """
    Borde condicional 2: Si el ítem es dudoso, no se encontró SKU o subió de precio >15%,
    se desvía a revisión humana. De lo contrario, se auto-homologa.
    """
    decision = state.get("decision_actual") or {}
    alerta = state.get("alerta_precio") or {}

    sku = decision.get("sku_seleccionado")
    confianza = decision.get("confianza", 0.0)
    requiere_revision = decision.get("requiere_revision", False)
    hay_alerta_precio = alerta.get("alerta", False)

    # Condiciones que exigen intervención humana
    if not sku or requiere_revision or confianza < 0.75 or hay_alerta_precio:
        return "intervencion_humana"
    return "aplicar_cambios"


def decidir_ruta_siguiente_item(state: AgentState) -> str:
    """
    Borde condicional 3 (Loop): Si quedan más ítems pendientes, cicla de regreso.
    Si se procesaron todos, avanza a guardar la auditoría.
    """
    pendientes = state.get("items_pendientes", [])
    if len(pendientes) > 0:
        return "preparar_siguiente_item"
    return "registrar_auditoria"


# ============================================================================
# 3. ENSAMBLAJE DEL GRAFO (LangGraph StateGraph)
# ============================================================================

def crear_grafo_agente():
    """
    Construye y compila el StateGraph del agente con soporte para MemorySaver (checkpointer).
    """
    builder = StateGraph(AgentState)

    # Registrar Nodos
    builder.add_node("extraer", node_extraer)
    builder.add_node("verificar_duplicidad", node_verificar_duplicidad)
    builder.add_node("manejar_duplicada", node_manejar_duplicada)
    builder.add_node("preparar_siguiente_item", node_preparar_siguiente_item)
    builder.add_node("recuperar_candidatos", node_recuperar_candidatos)
    builder.add_node("razonar_homologacion", node_razonar_homologacion)
    builder.add_node("evaluar_costo", node_evaluar_costo)
    builder.add_node("intervencion_humana", node_intervencion_humana)
    builder.add_node("aplicar_cambios", node_aplicar_cambios)
    builder.add_node("registrar_auditoria", node_registrar_auditoria)

    # Conectar Bordes y Flujo
    builder.add_edge(START, "extraer")
    builder.add_edge("extraer", "verificar_duplicidad")

    # Bifurcación 1: Duplicidad
    builder.add_conditional_edges(
        "verificar_duplicidad",
        decidir_ruta_duplicidad,
        {
            "manejar_duplicada": "manejar_duplicada",
            "preparar_siguiente_item": "preparar_siguiente_item"
        }
    )
    builder.add_edge("manejar_duplicada", END)

    # Flujo de procesamiento de ítem
    builder.add_edge("preparar_siguiente_item", "recuperar_candidatos")
    builder.add_edge("recuperar_candidatos", "razonar_homologacion")
    builder.add_edge("razonar_homologacion", "evaluar_costo")

    # Bifurcación 2: Decisión de revisión humana vs auto-escritura
    builder.add_conditional_edges(
        "evaluar_costo",
        decidir_ruta_evaluacion,
        {
            "intervencion_humana": "intervencion_humana",
            "aplicar_cambios": "aplicar_cambios"
        }
    )
    builder.add_edge("intervencion_humana", "aplicar_cambios")

    # Bifurcación 3: Loop de ítems restantes vs fin
    builder.add_conditional_edges(
        "aplicar_cambios",
        decidir_ruta_siguiente_item,
        {
            "preparar_siguiente_item": "preparar_siguiente_item",
            "registrar_auditoria": "registrar_auditoria"
        }
    )
    builder.add_edge("registrar_auditoria", END)

    # Configuración de memoria de corto plazo (Checkpointer)
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    return app

