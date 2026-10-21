"""
Orquestador de Flujo del Agente de Facturas con LangGraph.

Implementa una Máquina de Estados Finita (StateGraph) con:
- Nodos especializados que ejecutan las 9 herramientas creadas.
- Bordes condicionales para toma de decisiones adaptativas (duplicados, anomalías de precios, baja confianza).
- Ciclos de iteración (loops) para procesar múltiples ítems secuencialmente.
- Memoria de corto plazo (MemorySaver checkpointer) para dar soporte a Human-in-the-Loop.
"""
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from colorist import BgColor, green, yellow, red


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
# FUNCIONES AUXILIARES DE VALIDACIÓN DE LEGIBILIDAD
# ============================================================================

def _limpiar_texto(valor: Any) -> str:
    """Normaliza un campo de texto devolviendo cadena vacía si es None o 'null'."""
    if valor is None:
        return ""
    texto = str(valor).strip()
    if texto.lower() in ["null", "none", "n/a", "ilegible"]:
        return ""
    return texto


def _parsear_entero_positivo(valor: Any) -> Optional[int]:
    """Convierte un valor a entero positivo; retorna None si es ilegible, nulo o <= 0."""
    if valor is None:
        return None
    try:
        num = int(round(float(valor)))
        return num if num > 0 else None
    except (ValueError, TypeError):
        return None


def _parsear_float_positivo(valor: Any) -> Optional[float]:
    """Convierte un valor a float positivo; retorna None si es ilegible, nulo o <= 0."""
    if valor is None:
        return None
    try:
        num = float(valor)
        return num if num > 0 else None
    except (ValueError, TypeError):
        return None


def _obtener_campos_ilegibles(item: Dict[str, Any]) -> List[str]:
    """Identifica qué campos clave de una línea de factura son ilegibles o ausentes."""
    campos = []
    if not _limpiar_texto(item.get("nombre_glosa")):
        campos.append("nombre/glosa")
    if _parsear_entero_positivo(item.get("cantidad")) is None:
        campos.append("cantidad")
    if _parsear_float_positivo(item.get("precio_unitario")) is None:
        campos.append("precio unitario")
    return campos


# ============================================================================
# 1. NODOS DEL GRAFO (Execution Nodes)
# ============================================================================

def node_extraer(state):
    """
    Nodo 1: Invoca la herramienta multimodal para extraer encabezado e ítems desde la imagen.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo la ruta de la imagen.
    
    Returns:
        Dict[str, Any]: Estado actualizado con los datos extraídos de la factura y lista de ítems pendientes.
    """
    ruta_imagen = state["ruta_imagen"]
    print(f"\n{BgColor.YELLOW}[Agente - Percepción]{BgColor.OFF} Extrayendo datos de la factura: '{ruta_imagen}'...")

    datos_factura = extraer_factura_desde_imagen.invoke({"ruta_imagen": ruta_imagen}) or {}
    items = list(datos_factura.get("items") or [])

    msg = f"Factura extraída con éxito. Proveedor: '{datos_factura.get('proveedor')}', Folio: '{datos_factura.get('numero_factura')}', Ítems detectados: {len(items)}"
    print(f" -> {msg}")

    return {
        "datos_factura": datos_factura,
        "items_pendientes": items,
        "items_procesados": [],
        "mensajes_bitacora": state.get("mensajes_bitacora", []) + [msg],
        "estado_final": "EN_PROCESO"
    }


def node_verificar_duplicidad(state):
    """
    Nodo 2: Verifica en la memoria de largo plazo (Firestore) si la factura ya fue procesada.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo los datos de la factura extraída.
    
    Returns:
        Dict[str, Any]: Estado actualizado con la bandera de duplicidad y la información del registro existente si aplica.
    """
    datos = state["datos_factura"]
    folio = datos.get("numero_factura")
    proveedor = datos.get("proveedor")

    print(f"{BgColor.YELLOW}[Agente - Verificación]{BgColor.OFF} Comprobando duplicidad para folio '{folio}' de '{proveedor}'...")
    resultado = consultar_duplicidad_factura.invoke({"folio": folio, "proveedor": proveedor})

    es_duplicada = resultado.get("es_duplicada", False)
    msg = resultado.get("mensaje", "")
    print(f" -> {msg}")

    return {
        "es_duplicada": es_duplicada,
        "info_duplicada": resultado.get("registro_existente"),
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_manejar_duplicada(state):
    """
    Nodo Alternativo: Detiene el flujo adaptativamente si la factura ya existe,
    evitando la doble contabilización en inventario.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo la bandera de duplicidad y la información del registro existente.
        
    Returns:
        Dict[str, Any]: Estado final actualizado indicando que la factura fue ignorada por duplicidad.
    """
    info = state.get("info_duplicada") or {}
    msg = f"ADVERTENCIA: Proceso abortado para evitar duplicar stock. Factura folio '{state['datos_factura'].get('numero_factura')}' registrada el {info.get('fecha')}."
    print(f"\n{BgColor.YELLOW}[Agente - Control]{BgColor.OFF} {msg}")

    return {
        "estado_final": "DUPLICADO_IGNORADO",
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_preparar_siguiente_item(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 3 (Loop): Toma el primer ítem de la lista de pendientes para su evaluación individual.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo la lista de ítems pendientes.
        
    Returns:
        Dict[str, Any]: Estado actualizado con el ítem actual a evaluar y la lista de pendientes restante.
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


def node_recuperar_candidatos(state):
    """
    Nodo 4: Recuperación semántica de los top 3 candidatos más similares en Firestore.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo el ítem actual a evaluar.
        
    Returns:
        Dict[str, Any]: Estado actualizado con la lista de candidatos recuperados y mensaje de bitácora.
    """
    item = state["item_actual"]
    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))

    if not glosa:
        msg = "Nombre/glosa ilegible en la factura. Se omite búsqueda vectorial y se enviará a revisión humana."
        print(f"{BgColor.YELLOW}[Agente - Retrieval Semántico]{BgColor.OFF} -> {msg}")
        return {
            "candidatos_actuales": [],
            "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
        }

    print(f"{BgColor.YELLOW}[Agente - Retrieval Semántico]{BgColor.OFF} Buscando candidatos para '{glosa}'...")
    candidatos = buscar_candidatos_vectoriales.invoke({"glosa": glosa, "formato": formato, "top_k": 3})

    nombres_candidatos = [f"{c['sku']} ({c['nombre']})" for c in candidatos]
    msg = f"Candidatos recuperados: {', '.join(nombres_candidatos) if nombres_candidatos else 'Ninguno'}"
    print(f" -> {msg}")

    return {
        "candidatos_actuales": candidatos,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_razonar_homologacion(state):
    """
    Nodo 5: El LLM razona sobre abreviaciones y similitud con los candidatos recuperados.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo el ítem actual y los candidatos recuperados.
    
    Returns:
        Dict[str, Any]: Estado actualizado con la decisión de homologación del LLM y mensaje de bitácora.
    """
    item = state["item_actual"]
    candidatos = state["candidatos_actuales"]
    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))

    print(f"{BgColor.YELLOW}[Agente - Razonamiento LLM]{BgColor.OFF} Evaluando homologación...")

    if not glosa:
        decision = {
            "sku_seleccionado": None,
            "confianza": 0.0,
            "justificacion": "El nombre o glosa del producto no es legible en la factura.",
            "requiere_revision": True
        }
    else:
        decision = razonar_homologacion.invoke({
            "glosa_factura": glosa,
            "formato_factura": formato,
            "candidatos": candidatos
        }) or {}

    sku = decision.get("sku_seleccionado")
    confianza = float(decision.get("confianza") or 0.0)
    justificacion = decision.get("justificacion") or ""

    msg = f"Decisión de homologación: SKU={sku} (Confianza={confianza:.2f}) | Motivo: {justificacion}"
    print(f" -> {msg}")

    return {
        "decision_actual": decision,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg]
    }


def node_evaluar_costo(state):
    """
    Nodo 6: Evalúa si existe variación abrupta de costo en comparación al histórico.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo el ítem actual y la decisión de homologación.
    
    Returns:
        Dict[str, Any]: Estado actualizado con la alerta de variación de costo si aplica y mensaje de bitácora.
    """
    decision = state["decision_actual"] or {}
    item = state["item_actual"]
    sku = decision.get("sku_seleccionado")
    costo_factura = _parsear_float_positivo(item.get("precio_unitario"))

    alerta_precio = None
    if sku:
        if costo_factura is None:
            print(" -> [AVISO] Precio unitario ilegible o ausente en la factura; requerirá validación humana.")
        else:
            prod_info = consultar_producto_por_sku.invoke({"sku": sku})
            if prod_info.get("encontrado"):
                costo_actual = float(prod_info["producto"].get("precio_costo") or 0)
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


def node_intervencion_humana(state):
    """
    Nodo 7: Human-in-the-Loop.
    Se activa cuando:
    - La confianza es baja (<0.75) o el SKU es None (producto no existe).
    - El LLM marcó 'requiere_revision: true'.
    - Hubo una alerta crítica de costo (>15%).
    - Algún dato clave del ítem (nombre, cantidad, precio) no es legible en la factura.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo el ítem actual, decisión del LLM y alerta de costo.
    
    Returns:
        Dict[str, Any]: Estado actualizado con la intervención humana.
    """
    item = state["item_actual"]
    decision = state["decision_actual"] or {}
    alerta = state["alerta_precio"] or {}
    sku = decision.get("sku_seleccionado")
    confianza = float(decision.get("confianza") or 0.0)

    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))
    costo_factura = _parsear_float_positivo(item.get("precio_unitario"))
    cantidad = _parsear_entero_positivo(item.get("cantidad"))
    campos_ilegibles = _obtener_campos_ilegibles(item)

    print("\n" + "="*70)
    print(f" {BgColor.GREEN}[HUMAN-IN-THE-LOOP]{BgColor.OFF} Se requiere intervención humana para este ítem:")
    print(
        f" Glosa Factura: '{glosa or 'ILEGIBLE'}' | "
        f"Formato: {formato or 'No especificado/Ilegible'} | "
        f"Costo Factura: {f'${costo_factura:,.0f}' if costo_factura is not None else 'ILEGIBLE'} | "
        f"Cantidad: {cantidad if cantidad is not None else 'ILEGIBLE'}"
    )
    if campos_ilegibles:
        print(f" ⚠️ Campos ilegibles o incompletos detectados: {', '.join(campos_ilegibles)}")

    # CASO 1: Producto no encontrado o no homologable -> Dar de alta nuevo producto
    if not sku or decision.get("requiere_revision") or confianza < 0.75:
        print(f" Motivo: No hay coincidencia segura en el catálogo (Confianza: {confianza:.2f}).")
        print(f" Justificación: {decision.get('justificacion')}")

        resp = input(" -> ¿Deseas agregar este producto como NUEVO en el catálogo? (s/n): ").strip().lower()

        if resp in ["s", "si", "y", "yes"]:
            while not glosa:
                glosa = input("    * Nombre comercial (era ilegible en la factura): ").strip()
                if not glosa:
                    print("      Debes ingresar un nombre válido para el producto.")

            sku_sug = glosa.upper().replace(" ", "-")[:15]
            nuevo_sku = input(f"    * SKU único [Enter para '{sku_sug}']: ").strip().upper() or sku_sug
            nombre = input(f"    * Nombre comercial [Enter para '{glosa}']: ").strip() or glosa
            marca = input("    * Marca: ").strip() or "Genérico"
            categoria = input("    * Categoría (Destilados, Vinos, Cervezas, Abarrotes): ").strip() or "Otros"
            subcategoria = input("    * Subcategoría: ").strip() or "General"
            nuevo_formato = input(f"    * Formato [Enter para '{formato or 'Unidad'}']: ").strip() or (formato or "Unidad")

            while cantidad is None:
                cant_input = input("    * Cantidad comprada (era ilegible en la factura): ").strip()
                cantidad = _parsear_entero_positivo(cant_input)
                if cantidad is None:
                    print("      Ingresa una cantidad entera mayor a 0.")

            while costo_factura is None:
                costo_input = input("    * Precio de costo unitario (era ilegible en la factura): ").strip()
                costo_factura = _parsear_float_positivo(costo_input)
                if costo_factura is None:
                    print("      Ingresa un precio de costo válido mayor a 0.")

            while True:
                pv_input = input(f"    * Precio de venta al público (Costo factura: ${costo_factura:,.0f}): ").strip()
                precio_venta = _parsear_float_positivo(pv_input)
                if precio_venta is not None:
                    break
                print("      Ingresa un número válido mayor a 0.")

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

    # CASO 2: Producto homologado pero con datos ilegibles (cantidad/precio) y/o alerta de precio
    else:
        print(f" Producto homologado: {sku}")
        if campos_ilegibles:
            print(f" Motivo: Faltan datos legibles en la factura ({', '.join(campos_ilegibles)}).")
        if alerta.get("alerta"):
            print(f" Motivo: Alerta de variación de costo detectada ({alerta.get('variacion_porcentual')}%)")

        continuar = "s"
        if campos_ilegibles:
            continuar = input(" -> ¿Deseas completar los datos para actualizar este ítem? (s/n) [Enter='s']: ").strip().lower() or "s"

        if continuar not in ["s", "si", "y", "yes"]:
            print(" -> Ítem omitido por decisión del operador.")
            datos_accion = {"tipo_accion": "OMITIR", "sku": sku}
        else:
            while cantidad is None:
                cant_input = input("    * Ingresa la cantidad comprada (ilegible en factura): ").strip()
                cantidad = _parsear_entero_positivo(cant_input)
                if cantidad is None:
                    print("      Ingresa una cantidad entera mayor a 0.")

            while costo_factura is None:
                costo_input = input("    * Ingresa el precio de costo unitario (ilegible en factura): ").strip()
                costo_factura = _parsear_float_positivo(costo_input)
                if costo_factura is None:
                    print("      Ingresa un precio de costo válido mayor a 0.")

            pv_mod = input(" -> ¿Deseas ajustar el precio de venta? [Escribe nuevo precio o Enter para mantener]: ").strip()
            nuevo_pv = _parsear_float_positivo(pv_mod) if pv_mod else None

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


def node_aplicar_cambios(state):
    """
    Nodo 8: Aplica las escrituras atómicas en Firestore (actualizar stock o crear nuevo producto).
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo el ítem actual y la intervención humana si aplica.
        
    Returns:
        Dict[str, Any]: Estado actualizado con el registro del ítem procesado y mensaje de bitácora.
    """
    item = state["item_actual"]
    decision = state["decision_actual"] or {}
    accion = state.get("datos_intervencion_humana")

    registro_item = {
        "glosa_original": _limpiar_texto(item.get("nombre_glosa")) or "ILEGIBLE",
        "formato": _limpiar_texto(item.get("formato")) or None,
        "cantidad": _parsear_entero_positivo(item.get("cantidad")) or 0,
        "precio_unitario": _parsear_float_positivo(item.get("precio_unitario")) or 0,
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
            registro_item["glosa_original"] = registro_item["glosa_original"] if registro_item["glosa_original"] != "ILEGIBLE" else accion["nombre"]
            registro_item["cantidad"] = accion["stock_inicial"]
            registro_item["precio_unitario"] = accion["precio_costo"]
            registro_item["sku_homologado"] = accion["sku"]
            registro_item["estado"] = "CREADO_NUEVO"
            print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {resultado.get('mensaje')}")

        elif tipo == "ACTUALIZAR_EXISTENTE":
            resultado = actualizar_stock_y_costo.invoke({
                "sku": accion["sku"],
                "cantidad": accion["cantidad"],
                "nuevo_costo": accion["precio_costo"],
                "nuevo_precio_venta": accion.get("nuevo_precio_venta")
            })
            registro_item["cantidad"] = accion["cantidad"]
            registro_item["precio_unitario"] = accion["precio_costo"]
            registro_item["sku_homologado"] = accion["sku"]
            registro_item["estado"] = "ACTUALIZADO"
            print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {resultado.get('mensaje')}")

        else:
            registro_item["estado"] = "OMITIDO_POR_HUMANO"
            print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} Ítem marcado como omitido.")

    # Si fue homologado automáticamente sin intervención
    else:
        sku = decision.get("sku_seleccionado")
        cantidad = _parsear_entero_positivo(item.get("cantidad")) or 1
        costo = _parsear_float_positivo(item.get("precio_unitario")) or 0.0

        resultado = actualizar_stock_y_costo.invoke({
            "sku": sku,
            "cantidad": cantidad,
            "nuevo_costo": costo
        })
        registro_item["sku_homologado"] = sku
        registro_item["estado"] = "AUTO_HOMOLOGADO"
        print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {resultado.get('mensaje')}")

    items_actualizados = list(state.get("items_procesados", [])) + [registro_item]
    return {
        "items_procesados": items_actualizados
    }


def node_registrar_auditoria(state):
    """
    Nodo 9: Finaliza el flujo guardando el comprobante de auditoría en la colección
    'historial_facturas' de Firestore (Memoria transaccional de largo plazo).
    
    Args:
        state (AgentState): Estado final del agente, incluyendo los datos de la factura y los ítems procesados.
    
    Returns:
        Dict[str, Any]: Estado final actualizado con el ID de auditoría y mensaje de bitácora.
    """
    datos = state["datos_factura"]
    folio = _limpiar_texto(datos.get("numero_factura")) or "S/F"
    proveedor = _limpiar_texto(datos.get("proveedor")) or "DESCONOCIDO"
    items_procesados = state.get("items_procesados", [])

    print(f"\n{BgColor.YELLOW}[Agente - Auditoría]{BgColor.OFF} Guardando registro transaccional para folio '{folio}'...")

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

def decidir_ruta_duplicidad(state):
    """
    Borde condicional 1: Si la factura ya existe, salta al manejador de duplicados.
    Si es nueva (o no se pudo verificar duplicidad por falta de folio), arranca la iteración de productos.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo la bandera de duplicidad.
    
    Returns:
        str: Nombre del nodo siguiente según la condición evaluada.
    """
    if state.get("es_duplicada"):
        return "manejar_duplicada"
    if not state.get("items_pendientes"):
        return "registrar_auditoria"
    return "preparar_siguiente_item"


def decidir_ruta_evaluacion(state):
    """
    Borde condicional 2: Si el ítem es dudoso, no se encontró SKU, subió de precio >15%,
    o tiene valores ilegibles (nombre, cantidad, precio), se desvía a revisión humana (Human-in-the-Loop).
    De lo contrario, se auto-homologa.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo la decisión del LLM y la alerta de costo.
    
    Returns:
        str: Nombre del nodo siguiente según la condición evaluada.
    """
    item = state.get("item_actual") or {}
    decision = state.get("decision_actual") or {}
    alerta = state.get("alerta_precio") or {}

    sku = decision.get("sku_seleccionado")
    confianza = float(decision.get("confianza") or 0.0)
    requiere_revision = bool(decision.get("requiere_revision", False))
    hay_alerta_precio = bool(alerta.get("alerta", False))
    campos_ilegibles = _obtener_campos_ilegibles(item)

    # Condiciones que exigen intervención humana (incluyendo campos no legibles)
    if not sku or requiere_revision or confianza < 0.75 or hay_alerta_precio or len(campos_ilegibles) > 0:
        return "intervencion_humana"
    return "aplicar_cambios"


def decidir_ruta_siguiente_item(state):
    """
    Borde condicional 3 (Loop): Si quedan más ítems pendientes, cicla de regreso.
    Si se procesaron todos, avanza a guardar la auditoría.
    
    Args:
        state (AgentState): Estado actual del agente, incluyendo la lista de ítems pendientes.
    
    Returns:
        str: Nombre del nodo siguiente según la condición evaluada.
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
    
    Args:
        None

    Returns:
        StateGraph: El grafo del agente compilado.
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
            "preparar_siguiente_item": "preparar_siguiente_item",
            "registrar_auditoria": "registrar_auditoria"
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

    return app