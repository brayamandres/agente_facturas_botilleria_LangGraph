"""
Orquestador de Flujo del Agente de Facturas con LangGraph.

Implementa una Máquina de Estados Finita (StateGraph) con:
- Nodos especializados que ejecutan las 9 herramientas creadas.
- Bordes condicionales para toma de decisiones adaptativas (duplicados, anomalías de precios, baja confianza, datos ilegibles).
- Ciclos de iteración (loops) para procesar múltiples ítems secuencialmente.
- Persistencia real de corto/mediano plazo en SQLite (SqliteSaver) y pausa/reanudación nativa
  mediante interrupt() y Command(resume=...) para continuidad efectiva ante interrupciones (IE3 e IE4).
"""
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from colorist import BgColor, green, yellow, red

from src.agent.state import AgentState
from src.services.checkpointer import get_sqlite_checkpointer
from src.tools.reasoning_tools import validar_respuesta_homologacion
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
# FUNCIONES AUXILIARES DE VALIDACIÓN DE LEGIBILIDAD, UNIDADES E HITL
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


def construir_payload_intervencion(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye el payload estructurado y serializable que se envía a interrupt()
    cuando un ítem requiere intervención humana (Human-in-the-Loop).
    """
    item = state.get("item_actual") or {}
    decision = state.get("decision_actual") or {}
    alerta = state.get("alerta_precio") or {}
    sku = decision.get("sku_seleccionado")
    confianza = float(decision.get("confianza") or 0.0)

    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))
    costo_factura = _parsear_float_positivo(item.get("precio_unitario"))
    cantidad = _parsear_entero_positivo(item.get("cantidad"))
    unidades_pack = _parsear_entero_positivo(item.get("unidades")) or 1
    campos_ilegibles = _obtener_campos_ilegibles(item)

    sin_coincidencia = bool(not sku or decision.get("requiere_revision") or confianza < 0.75)

    return {
        "tipo_intervencion": "ALTA_O_REVISION" if sin_coincidencia else "VALIDAR_EXISTENTE",
        "glosa": glosa,
        "formato": formato,
        "costo_factura": costo_factura,
        "cantidad": cantidad,
        "unidades_pack": unidades_pack,
        "campos_ilegibles": campos_ilegibles,
        "sku_homologado": sku,
        "confianza": confianza,
        "justificacion": decision.get("justificacion") or "",
        "alerta_precio": alerta,
        "candidatos": list(state.get("candidatos_actuales") or []),
    }


def solicitar_decision_humana_consola(
    payload: Dict[str, Any],
    permitir_pausa: bool = True,
) -> Dict[str, Any]:
    """
    Interfaz interactiva de consola para capturar la decisión del operador ante un interrupt()
    de LangGraph. Si 'permitir_pausa' es True, el operador puede pausar la factura ('p' o Ctrl+C)
    dejando el estado persistido en SQLite para retomarlo incluso después de cerrar el programa.
    """
    glosa = payload.get("glosa") or ""
    formato = payload.get("formato") or ""
    costo_factura = payload.get("costo_factura")
    cantidad = payload.get("cantidad")
    unidades_pack = int(payload.get("unidades_pack") or 1)
    campos_ilegibles = list(payload.get("campos_ilegibles") or [])
    sku = payload.get("sku_homologado")
    confianza = float(payload.get("confianza") or 0.0)
    justificacion = payload.get("justificacion") or ""
    alerta = payload.get("alerta_precio") or {}
    candidatos = list(payload.get("candidatos") or [])
    tipo_intervencion = payload.get("tipo_intervencion", "ALTA_O_REVISION")

    print("\n" + "=" * 75)
    print(f" {BgColor.GREEN}[HUMAN-IN-THE-LOOP - PAUSA LANGGRAPH]{BgColor.OFF} Se requiere decisión del operador:")
    print(
        f" Glosa Factura: '{glosa or 'ILEGIBLE'}' | "
        f"Formato: {formato or 'No especificado/Ilegible'} | "
        f"Costo Factura: {f'${costo_factura:,.0f}' if costo_factura is not None else 'ILEGIBLE'} | "
        f"Cantidad: {cantidad if cantidad is not None else 'ILEGIBLE'} (Unidades x pack: {unidades_pack})"
    )
    if campos_ilegibles:
        print(f" ⚠️ Campos ilegibles o incompletos detectados: {', '.join(campos_ilegibles)}")

    try:
        # CASO 1: Producto no encontrado o baja confianza -> Crear nuevo, homologar manual, omitir o pausar
        if tipo_intervencion == "ALTA_O_REVISION":
            print(f" Motivo: No hay coincidencia segura en el catálogo (Confianza: {confianza:.2f}).")
            print(f" Justificación: {justificacion}")

            if candidatos:
                print(" Candidatos cercanos en catálogo:")
                for idx_c, cand in enumerate(candidatos, start=1):
                    print(
                        f"   [{idx_c}] {cand.get('sku')} - {cand.get('nombre')} "
                        f"({cand.get('formato')}) | Costo actual: ${cand.get('precio_costo_actual', 0):,.0f}"
                    )

            opciones_txt = "(s=Crear nuevo / h=Usar candidato existente / n=Omitir"
            if permitir_pausa:
                opciones_txt += " / p=Pausar y guardar sesión)"
            else:
                opciones_txt += ")"

            resp = input(f"\n -> ¿Qué acción deseas realizar? {opciones_txt}: ").strip().lower()

            if permitir_pausa and resp in ["p", "pausar", "pause"]:
                print("=" * 75 + "\n")
                return {"tipo_accion": "PAUSAR_SESION"}

            if resp == "h" and candidatos:
                sel_c = input(f"    * Selecciona número de candidato (1-{len(candidatos)}) o escribe el SKU: ").strip()
                if sel_c.isdigit() and 1 <= int(sel_c) <= len(candidatos):
                    sku_elegido = candidatos[int(sel_c) - 1]["sku"]
                else:
                    sku_elegido = sel_c.upper()

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

                pv_mod = input("    * ¿Ajustar precio de venta? [Nuevo precio o Enter para mantener]: ").strip()
                nuevo_pv = _parsear_float_positivo(pv_mod) if pv_mod else None

                datos_accion = {
                    "tipo_accion": "ACTUALIZAR_EXISTENTE",
                    "sku": sku_elegido,
                    "cantidad": cantidad * unidades_pack,
                    "precio_costo": costo_factura,
                    "nuevo_precio_venta": nuevo_pv,
                    "unidades_incluidas": True,
                }

            elif resp in ["s", "si", "sí", "y", "yes"]:
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
                    "stock_inicial": cantidad * unidades_pack,
                    "unidades_incluidas": True,
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

            opciones_txt = "(s=Actualizar [Enter] / n=Omitir"
            if permitir_pausa:
                opciones_txt += " / p=Pausar y guardar sesión)"
            else:
                opciones_txt += ")"

            continuar = input(f"\n -> ¿Qué acción deseas realizar? {opciones_txt}: ").strip().lower() or "s"

            if permitir_pausa and continuar in ["p", "pausar", "pause"]:
                print("=" * 75 + "\n")
                return {"tipo_accion": "PAUSAR_SESION"}

            if continuar not in ["s", "si", "sí", "y", "yes"]:
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
                    "cantidad": cantidad * unidades_pack,
                    "precio_costo": costo_factura,
                    "nuevo_precio_venta": nuevo_pv,
                    "unidades_incluidas": True,
                }

    except (KeyboardInterrupt, EOFError):
        if permitir_pausa:
            print("\n -> Interrupción detectada: la sesión queda pausada y guardada en SQLite.")
            print("=" * 75 + "\n")
            return {"tipo_accion": "PAUSAR_SESION"}
        raise

    print("=" * 75 + "\n")
    return datos_accion


# ============================================================================
# 1. NODOS DEL GRAFO (Execution Nodes)
# ============================================================================

def node_extraer(state):
    """
    Nodo 1: Invoca la herramienta multimodal para extraer encabezado e ítems desde la imagen.
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
        "estado_final": "EN_PROCESO",
    }


def node_verificar_duplicidad(state):
    """
    Nodo 2: Verifica en la memoria de largo plazo (Firestore) si la factura ya fue procesada.
    """
    datos = state["datos_factura"]
    folio = datos.get("numero_factura")
    proveedor = datos.get("proveedor")

    print(f"{BgColor.YELLOW}[Agente - Verificación]{BgColor.OFF} Comprobando duplicidad para folio '{folio}' de '{proveedor}'...")
    resultado = consultar_duplicidad_factura.invoke({"folio": folio, "proveedor": proveedor})

    es_duplicada = bool(resultado.get("es_duplicada", False))
    estado_verificacion = resultado.get("estado_verificacion", "")
    msg = resultado.get("mensaje", "")
    print(f" -> {msg}")

    actualizacion = {
        "es_duplicada": es_duplicada,
        "info_duplicada": resultado.get("registro_existente"),
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
    }
    if estado_verificacion == "ERROR_CONSULTA" or resultado.get("es_duplicada") is None:
        actualizacion["estado_final"] = "ERROR"

    return actualizacion


def node_manejar_duplicada(state):
    """
    Nodo Alternativo: Detiene el flujo adaptativamente si la factura ya existe
    o si ocurrió un error de conexión/consulta al verificar duplicidad en Firestore.
    """
    if state.get("estado_final") == "ERROR":
        msg = "ERROR: Proceso abortado porque falló la consulta de verificación de duplicidad en Firestore."
        print(f"\n{BgColor.RED}[Agente - Error Control]{BgColor.OFF} {msg}")
        return {
            "estado_final": "ERROR",
            "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
        }

    info = state.get("info_duplicada") or {}
    msg = f"ADVERTENCIA: Proceso abortado para evitar duplicar stock. Factura folio '{state['datos_factura'].get('numero_factura')}' registrada el {info.get('fecha')}."
    print(f"\n{BgColor.YELLOW}[Agente - Control]{BgColor.OFF} {msg}")

    return {
        "estado_final": "DUPLICADO_IGNORADO",
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
    }


def node_preparar_siguiente_item(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 3 (Loop): Toma el primer ítem de la lista de pendientes para su evaluación individual.
    """
    pendientes = state["items_pendientes"]
    item_actual = pendientes[0]
    restantes = pendientes[1:]

    unidades_pack = _parsear_entero_positivo(item_actual.get("unidades")) or 1
    msg = (
        f"Iniciando evaluación de ítem: '{item_actual.get('nombre_glosa')}' "
        f"(Formato: {item_actual.get('formato')}, Cantidad: {item_actual.get('cantidad')}, Unidades/Pack: {unidades_pack})"
    )
    print(f"\n--- {msg} ---")

    return {
        "item_actual": item_actual,
        "items_pendientes": restantes,
        "candidatos_actuales": [],
        "decision_actual": None,
        "alerta_precio": None,
        "requiere_intervencion": False,
        "datos_intervencion_humana": None,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
    }


def node_recuperar_candidatos(state):
    """
    Nodo 4: Recuperación semántica de los top 3 candidatos más similares en Firestore.
    """
    item = state["item_actual"]
    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))

    if not glosa:
        msg = "Nombre/glosa ilegible en la factura. Se omite búsqueda vectorial y se enviará a revisión humana."
        print(f"{BgColor.YELLOW}[Agente - Retrieval Semántico]{BgColor.OFF} -> {msg}")
        return {
            "candidatos_actuales": [],
            "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
        }

    print(f"{BgColor.YELLOW}[Agente - Retrieval Semántico]{BgColor.OFF} Buscando candidatos para '{glosa}'...")
    candidatos = buscar_candidatos_vectoriales.invoke({"glosa": glosa, "formato": formato, "top_k": 3})

    nombres_candidatos = [f"{c['sku']} ({c['nombre']})" for c in candidatos]
    msg = f"Candidatos recuperados: {', '.join(nombres_candidatos) if nombres_candidatos else 'Ninguno'}"
    print(f" -> {msg}")

    return {
        "candidatos_actuales": candidatos,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
    }


def node_razonar_homologacion(state):
    """
    Nodo 5: El LLM razona sobre abreviaciones y similitud con los candidatos recuperados.
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
            "requiere_revision": True,
        }
    else:
        decision_cruda = razonar_homologacion.invoke({
            "glosa_factura": glosa,
            "formato_factura": formato,
            "candidatos": candidatos,
        }) or {}
        decision = validar_respuesta_homologacion(decision_cruda, candidatos=candidatos)

    sku = decision.get("sku_seleccionado")
    confianza = float(decision.get("confianza") or 0.0)
    justificacion = decision.get("justificacion") or ""

    msg = f"Decisión de homologación: SKU={sku} (Confianza={confianza:.2f}) | Motivo: {justificacion}"
    print(f" -> {msg}")

    return {
        "decision_actual": decision,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
    }


def node_evaluar_costo(state):
    """
    Nodo 6: Evalúa si existe variación abrupta de costo en comparación al histórico.
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
                    "umbral_alerta": 0.15,
                })

                if alerta_precio.get("alerta"):
                    print(f" -> [ALERTA DE COSTO] {alerta_precio.get('mensaje')}")

    return {
        "alerta_precio": alerta_precio,
    }


def node_intervencion_humana(state):
    """
    Nodo 7: Human-in-the-Loop mediante el mecanismo nativo interrupt() de LangGraph.

    Funcionamiento:
    - En la primera ejecución del nodo, interrupt(payload) suspende el grafo y persiste
      el checkpoint completo en SQLite bajo el thread_id activo.
    - Al reanudar la tarea (en la misma ejecución o tras cerrar y reabrir el programa)
      mediante grafo.invoke(Command(resume=datos_accion), config=config), interrupt(payload)
      retorna directamente 'datos_accion' y avanza a 'aplicar_cambios'.
    """
    payload = construir_payload_intervencion(state)

    try:
        datos_accion = interrupt(payload)
    except RuntimeError:
        # Respaldo únicamente cuando el nodo se prueba como función directa fuera de grafo.invoke()
        datos_accion = solicitar_decision_humana_consola(payload, permitir_pausa=False)

    tipo = (datos_accion or {}).get("tipo_accion", "OMITIR")
    sku_res = (datos_accion or {}).get("sku")
    msg = f"Decisión humana aplicada: acción='{tipo}', SKU='{sku_res}'."

    return {
        "datos_intervencion_humana": datos_accion,
        "requiere_intervencion": False,
        "mensajes_bitacora": list(state.get("mensajes_bitacora", [])) + [msg],
    }


def node_aplicar_cambios(state):
    """
    Nodo 8: Aplica las escrituras en Firestore (actualizar stock o crear nuevo producto).
    Multiplica 'cantidad' por 'unidades' (tamaño del pack) para sumar el total de unidades individuales
    y verifica 'resultado["exito"]' antes de registrar el estado del ítem.
    """
    item = state["item_actual"]
    decision = state["decision_actual"] or {}
    accion = state.get("datos_intervencion_humana")
    folio_actual = _limpiar_texto((state.get("datos_factura") or {}).get("numero_factura"))

    cantidad_bultos = _parsear_entero_positivo(item.get("cantidad")) or 0
    unidades_pack = _parsear_entero_positivo(item.get("unidades")) or 1
    unidades_totales = cantidad_bultos * unidades_pack

    registro_item = {
        "glosa_original": _limpiar_texto(item.get("nombre_glosa")) or "ILEGIBLE",
        "formato": _limpiar_texto(item.get("formato")) or None,
        "cantidad": unidades_totales,
        "unidades_por_pack": unidades_pack,
        "precio_unitario": _parsear_float_positivo(item.get("precio_unitario")) or 0,
        "sku_homologado": None,
        "estado": "PENDIENTE",
    }
    msg = ""

    # Si vino de intervención humana
    if accion:
        tipo = accion.get("tipo_accion")
        if tipo == "CREAR_NUEVO":
            stock_inicial = accion["stock_inicial"]
            if not accion.get("unidades_incluidas"):
                stock_inicial = stock_inicial * unidades_pack

            args_crear = {
                "sku": accion["sku"],
                "nombre": accion["nombre"],
                "marca": accion["marca"],
                "categoria": accion["categoria"],
                "subcategoria": accion["subcategoria"],
                "formato": accion["formato"],
                "precio_costo": accion["precio_costo"],
                "precio_venta": accion["precio_venta"],
                "stock_inicial": stock_inicial,
            }
            if folio_actual:
                args_crear["folio_factura"] = folio_actual

            resultado = registrar_nuevo_producto.invoke(args_crear) or {}

            msg = resultado.get("mensaje", "")
            registro_item["glosa_original"] = (
                registro_item["glosa_original"]
                if registro_item["glosa_original"] != "ILEGIBLE"
                else accion["nombre"]
            )
            registro_item["cantidad"] = stock_inicial
            registro_item["precio_unitario"] = accion["precio_costo"]
            registro_item["sku_homologado"] = accion["sku"]

            if resultado.get("exito"):
                registro_item["estado"] = "CREADO_NUEVO"
                print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")
            else:
                registro_item["estado"] = "ERROR_ESCRITURA"
                print(f"{BgColor.RED}[Agente - Error Escritura]{BgColor.OFF} {msg}")

        elif tipo == "ACTUALIZAR_EXISTENTE":
            cantidad_actualizar = accion["cantidad"]
            if not accion.get("unidades_incluidas"):
                cantidad_actualizar = cantidad_actualizar * unidades_pack

            args_act = {
                "sku": accion["sku"],
                "cantidad": cantidad_actualizar,
                "nuevo_costo": accion["precio_costo"],
                "nuevo_precio_venta": accion.get("nuevo_precio_venta"),
            }
            if folio_actual:
                args_act["folio_factura"] = folio_actual

            resultado = actualizar_stock_y_costo.invoke(args_act) or {}

            msg = resultado.get("mensaje", "")
            registro_item["cantidad"] = cantidad_actualizar
            registro_item["precio_unitario"] = accion["precio_costo"]
            registro_item["sku_homologado"] = accion["sku"]

            if resultado.get("exito"):
                registro_item["estado"] = "ACTUALIZADO"
                print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")
            else:
                registro_item["estado"] = "ERROR_ESCRITURA"
                print(f"{BgColor.RED}[Agente - Error Escritura]{BgColor.OFF} {msg}")

        else:
            registro_item["estado"] = "OMITIDO_POR_HUMANO"
            msg = "Ítem marcado como omitido."
            print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")

    # Si fue homologado automáticamente sin intervención
    else:
        sku = decision.get("sku_seleccionado")
        cantidad_base = _parsear_entero_positivo(item.get("cantidad")) or 1
        cantidad_total = cantidad_base * unidades_pack
        costo = _parsear_float_positivo(item.get("precio_unitario")) or 0.0

        args_auto = {
            "sku": sku,
            "cantidad": cantidad_total,
            "nuevo_costo": costo,
        }
        if folio_actual:
            args_auto["folio_factura"] = folio_actual

        resultado = actualizar_stock_y_costo.invoke(args_auto) or {}

        msg = resultado.get("mensaje", "")
        registro_item["cantidad"] = cantidad_total
        registro_item["sku_homologado"] = sku

        if resultado.get("exito"):
            registro_item["estado"] = "AUTO_HOMOLOGADO"
            print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")
        else:
            registro_item["estado"] = "ERROR_ESCRITURA"
            print(f"{BgColor.RED}[Agente - Error Escritura]{BgColor.OFF} {msg}")

    items_actualizados = list(state.get("items_procesados", [])) + [registro_item]
    bitacora = list(state.get("mensajes_bitacora", [])) + ([msg] if msg else [])
    return {
        "items_procesados": items_actualizados,
        "mensajes_bitacora": bitacora,
    }


def node_registrar_auditoria(state):
    """
    Nodo 9: Finaliza el flujo guardando el comprobante de auditoría en la colección
    'historial_facturas' de Firestore (Memoria transaccional de largo plazo).
    """
    datos = state["datos_factura"]
    folio = _limpiar_texto(datos.get("numero_factura")) or "S/F"
    proveedor = _limpiar_texto(datos.get("proveedor")) or "DESCONOCIDO"
    items_procesados = state.get("items_procesados", [])

    hubo_errores = any(i.get("estado") == "ERROR_ESCRITURA" for i in items_procesados)
    estado_auditoria = "COMPLETADO_CON_ERRORES" if hubo_errores else "COMPLETADO"

    print(f"\n{BgColor.YELLOW}[Agente - Auditoría]{BgColor.OFF} Guardando registro transaccional para folio '{folio}'...")

    resultado = registrar_auditoria_factura.invoke({
        "folio": folio,
        "proveedor": proveedor,
        "total_items": len(items_procesados),
        "items_procesados": items_procesados,
        "estado": estado_auditoria,
    })

    id_auditoria = resultado.get("id_auditoria")
    msg = f"Factura folio '{folio}' procesada ({estado_auditoria}). Auditoría ID: {id_auditoria}"
    print(f" -> {msg}\n")

    return {
        "auditoria_id": id_auditoria,
        "estado_final": estado_auditoria,
        "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
    }


# ============================================================================
# 2. BORDES CONDICIONALES (Adaptive Decision Edges)
# ============================================================================

def decidir_ruta_duplicidad(state):
    """
    Borde condicional 1: Si la factura ya existe o hubo un error al consultar Firestore,
    salta al manejador para detener el proceso.
    Si es nueva (o el folio era ilegible/ausente), arranca la iteración de productos.
    """
    if state.get("es_duplicada") or state.get("estado_final") == "ERROR":
        return "manejar_duplicada"
    if not state.get("items_pendientes"):
        return "registrar_auditoria"
    return "preparar_siguiente_item"


def decidir_ruta_evaluacion(state):
    """
    Borde condicional 2: Si el ítem es dudoso, no se encontró SKU, subió de precio >15%,
    o tiene valores ilegibles (nombre, cantidad, precio), se desvía a revisión humana (Human-in-the-Loop).
    De lo contrario, se auto-homologa.
    """
    item = state.get("item_actual") or {}
    decision = state.get("decision_actual") or {}
    alerta = state.get("alerta_precio") or {}

    sku = decision.get("sku_seleccionado")
    confianza = float(decision.get("confianza") or 0.0)
    requiere_revision = bool(decision.get("requiere_revision", False))
    hay_alerta_precio = bool(alerta.get("alerta", False))
    campos_ilegibles = _obtener_campos_ilegibles(item)

    if not sku or requiere_revision or confianza < 0.75 or hay_alerta_precio or len(campos_ilegibles) > 0:
        return "intervencion_humana"
    return "aplicar_cambios"


def decidir_ruta_siguiente_item(state):
    """
    Borde condicional 3 (Loop): Si quedan más ítems pendientes, cicla de regreso.
    Si se procesaron todos, avanza a guardar la auditoría.
    """
    pendientes = state.get("items_pendientes", [])
    if len(pendientes) > 0:
        return "preparar_siguiente_item"
    return "registrar_auditoria"


# ============================================================================
# 3. ENSAMBLAJE DEL GRAFO (LangGraph StateGraph con Checkpointer SQLite)
# ============================================================================

_grafo_compilado_singleton = None


def crear_grafo_agente(checkpointer=None):
    """
    Construye y compila el StateGraph del agente con soporte para checkpointer persistente en SQLite.

    Args:
        checkpointer (BaseCheckpointSaver | None): Checkpointer opcional (por defecto usa SQLite en 'data/checkpoints.sqlite').

    Returns:
        CompiledStateGraph: El grafo del agente compilado.
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
            "registrar_auditoria": "registrar_auditoria",
        },
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
            "aplicar_cambios": "aplicar_cambios",
        },
    )
    builder.add_edge("intervencion_humana", "aplicar_cambios")

    # Bifurcación 3: Loop de ítems restantes vs fin
    builder.add_conditional_edges(
        "aplicar_cambios",
        decidir_ruta_siguiente_item,
        {
            "preparar_siguiente_item": "preparar_siguiente_item",
            "registrar_auditoria": "registrar_auditoria",
        },
    )
    builder.add_edge("registrar_auditoria", END)

    # Configuración de persistencia de estado en SQLite
    saver = checkpointer if checkpointer is not None else get_sqlite_checkpointer()
    app = builder.compile(checkpointer=saver)

    return app


def obtener_grafo_agente():
    """
    Retorna la instancia Singleton del grafo compilado con checkpointer SQLite,
    evitando reconstruir el grafo en cada operación del menú.
    """
    global _grafo_compilado_singleton
    if _grafo_compilado_singleton is None:
        _grafo_compilado_singleton = crear_grafo_agente()
    return _grafo_compilado_singleton
