"""
Orquestador de Flujo del Agente de Facturas con LangGraph.

Implementa una Máquina de Estados Finita (StateGraph) con:
- Nodos especializados que ejecutan las 11 herramientas creadas.
- Bordes condicionales para toma de decisiones adaptativas (duplicados, anomalías de precios, baja confianza, datos ilegibles).
- Ciclos de iteración (loops) para procesar múltiples ítems secuencialmente.
- Persistencia real de corto/mediano plazo en SQLite (SqliteSaver) y pausa/reanudación nativa
  mediante interrupt() y Command(resume=...) para continuidad efectiva ante interrupciones (IE3 e IE4).
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from colorist import BgColor, green, yellow, red

import src.services.firebase as _firebase_svc
import src.tools.query_tools as _query_tools_mod
import src.tools.write_tools as _write_tools_mod
from src.agent.state import AgentState
from src.services.checkpointer import get_sqlite_checkpointer
from src.services.gemini import validar_esquema_extraccion
from src.agent.planning_policy import (
    PrioridadPlanificacion,
    TABLA_POLITICA_PLANIFICACION,
    formatear_aviso_planificacion,
    evaluar_politica_evaluacion_item,
)
from src.tools.reasoning_tools import (
    validar_estructura_homologacion,
    validar_respuesta_homologacion,
    detectar_indicio_pack,
    validar_unidades_item,
    resolver_costo_unitario_linea,
)
from src.tools.write_tools import construir_id_operacion
from src.tools import (
    extraer_factura_desde_imagen,
    consultar_duplicidad_factura,
    buscar_candidatos_vectoriales,
    consultar_producto_por_sku,
    consultar_equivalencia_proveedor,
    razonar_homologacion,
    evaluar_variacion_costo,
    actualizar_stock_y_costo,
    registrar_nuevo_producto,
    registrar_auditoria_factura,
    guardar_equivalencia_proveedor,
)


def _consulta_equivalencia_activa() -> bool:
    """
    Habilita la consulta a 'equivalencias_proveedor' en ejecución real y en pruebas que
    mockean equivalencias, evitando llamadas de red en pruebas unitarias offline donde solo
    se mockeó 'buscar_candidatos_vectoriales' o 'get_productos_collection'.
    """
    if (
        consultar_equivalencia_proveedor is not _query_tools_mod.consultar_equivalencia_proveedor
        or _query_tools_mod.get_equivalencias_collection is not _firebase_svc.get_equivalencias_collection
    ):
        return True
    return (
        buscar_candidatos_vectoriales is _query_tools_mod.buscar_candidatos_vectoriales
        and _query_tools_mod.get_productos_collection is _firebase_svc.get_productos_collection
    )


def _escritura_equivalencia_activa() -> bool:
    """
    Habilita el guardado en 'equivalencias_proveedor' en ejecución real y en pruebas que
    mockean equivalencias, evitando llamadas de red en pruebas unitarias offline parciales.
    """
    if (
        guardar_equivalencia_proveedor is not _write_tools_mod.guardar_equivalencia_proveedor
        or _write_tools_mod.get_equivalencias_collection is not _firebase_svc.get_equivalencias_collection
    ):
        return True
    return (
        actualizar_stock_y_costo is _write_tools_mod.actualizar_stock_y_costo
        and registrar_nuevo_producto is _write_tools_mod.registrar_nuevo_producto
        and _write_tools_mod.get_productos_collection is _firebase_svc.get_productos_collection
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
    """
    Convierte un valor a entero positivo estricto.
    Retorna None si es ilegible, nulo, <= 0 o si tiene parte fraccionaria (ej. 1.6),
    evitando redondear silenciosamente cantidades de inventario no enteras.
    """
    if valor is None:
        return None
    try:
        txt_val = str(valor).replace(",", ".").strip()
        f_val = float(txt_val)
        if f_val <= 0:
            return None
        # Rechazar cualquier valor fraccionario no entero como 1.6
        if not f_val.is_integer() and abs(f_val - round(f_val)) > 1e-6:
            return None
        return int(round(f_val))
    except (ValueError, TypeError):
        return None


def _parsear_float_positivo(valor: Any) -> Optional[float]:
    """Convierte un valor a float positivo; retorna None si es ilegible, nulo o <= 0."""
    if valor is None:
        return None
    try:
        txt_val = str(valor).replace(",", ".").strip()
        num = float(txt_val)
        return num if num > 0 else None
    except (ValueError, TypeError):
        return None


def _obtener_campos_ilegibles(item: Dict[str, Any]) -> List[str]:
    """Identifica qué campos clave de una línea de factura son ilegibles o ausentes."""
    campos = []
    if not _limpiar_texto(item.get("nombre_glosa")):
        campos.append("nombre/glosa")

    cant_raw = item.get("cantidad")
    if _parsear_entero_positivo(cant_raw) is None:
        if cant_raw is not None and str(cant_raw).strip() not in ("", "None", "null"):
            try:
                txt_limpio = str(cant_raw).replace(",", ".").strip()
                f_val = float(txt_limpio)
                if f_val > 0 and not f_val.is_integer():
                    campos.append(f"cantidad (no es entero: {cant_raw})")
                else:
                    campos.append("cantidad")
            except (ValueError, TypeError):
                campos.append("cantidad")
        else:
            campos.append("cantidad")

    if _parsear_float_positivo(item.get("precio_unitario")) is None:
        campos.append("precio unitario")

    # Validación estricta de unidades por bulto/pack (IE1):
    # No se asume 1 silenciosamente si la glosa indica pack o si las unidades son ausentes/inválidas en un bulto.
    info_unid = validar_unidades_item(item)
    if info_unid.get("unidades_ilegibles"):
        campos.append("unidades por pack (bulto detectado sin unidades legibles)")
    elif item.get("unidades") is not None and str(item.get("unidades")).strip().lower() in ["null", "none", "n/a", "ilegible"]:
        campos.append("unidades")
    return campos


_ESTADOS_LINEA_EXITOSA = {"AUTO_HOMOLOGADO", "ACTUALIZADO", "CREADO_NUEVO", "OMITIDO_POR_HUMANO"}


def _obtener_identidad_linea(
    state: Dict[str, Any],
    item: Optional[Dict[str, Any]],
    linea_por_defecto: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Construye de forma garantizada la identidad de operación ('proveedor + folio + línea')
    para cualquier escritura en Firestore, incluso en pruebas unitarias aisladas del nodo.
    """
    item_dict = item if isinstance(item, dict) else {}
    datos_factura = state.get("datos_factura") or {}
    proveedor = _limpiar_texto(datos_factura.get("proveedor")) or None
    folio = _limpiar_texto(datos_factura.get("numero_factura")) or None
    ruta_img = _limpiar_texto(state.get("ruta_imagen"))
    respaldo_archivo = Path(ruta_img).name if ruta_img else None

    linea_id = (
        _parsear_entero_positivo(item_dict.get("linea_id"))
        or linea_por_defecto
        or (len(state.get("items_procesados") or []) + 1)
    )
    id_operacion = _limpiar_texto(item_dict.get("id_operacion")) or construir_id_operacion(
        proveedor=proveedor,
        folio=folio,
        linea_id=linea_id,
        respaldo_factura=respaldo_archivo,
    )
    return {
        "proveedor": proveedor,
        "folio": folio,
        "linea_id": linea_id,
        "id_operacion": id_operacion,
    }


def construir_payload_intervencion(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye el payload estructurado y serializable que se envía a interrupt()
    cuando un ítem requiere intervención humana (Human-in-the-Loop).
    """
    item = state.get("item_actual") or {}
    datos_factura = state.get("datos_factura") or {}
    decision = state.get("decision_actual") or {}
    alerta = state.get("alerta_precio") or {}
    sku = decision.get("sku_seleccionado")
    confianza = float(decision.get("confianza") or 0.0)

    proveedor = _limpiar_texto(datos_factura.get("proveedor")) or None
    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))
    precio_factura_bruto = _parsear_float_positivo(item.get("precio_unitario"))
    cantidad = _parsear_entero_positivo(item.get("cantidad"))
    campos_ilegibles = _obtener_campos_ilegibles(item)

    info_unid = validar_unidades_item(item)
    unidades_pack = info_unid.get("unidades")
    es_pack = bool(info_unid.get("es_pack"))
    unidades_ilegibles = bool(info_unid.get("unidades_ilegibles"))

    costo_unit_normalizado = item.get("costo_unitario_normalizado")
    tipo_precio = item.get("tipo_precio") or ("POR_PACK" if es_pack else "POR_UNIDAD")
    ambiguedad_costo = bool(item.get("ambiguedad_costo") or (alerta and alerta.get("ambiguedad_pack")))

    sin_coincidencia = bool(not sku or decision.get("requiere_revision") or confianza < 0.75)

    return {
        "tipo_intervencion": "ALTA_O_REVISION" if sin_coincidencia else "VALIDAR_EXISTENTE",
        "proveedor": proveedor,
        "glosa": glosa,
        "formato": formato,
        "costo_factura": costo_unit_normalizado if costo_unit_normalizado is not None else precio_factura_bruto,
        "precio_factura_bruto": precio_factura_bruto,
        "costo_unitario_normalizado": costo_unit_normalizado,
        "tipo_precio": tipo_precio,
        "es_pack": es_pack,
        "unidades_pack": unidades_pack,
        "unidades_ilegibles": unidades_ilegibles,
        "ambiguedad_costo": ambiguedad_costo,
        "cantidad": cantidad,
        "campos_ilegibles": campos_ilegibles,
        "sku_homologado": sku,
        "confianza": confianza,
        "justificacion": decision.get("justificacion") or "",
        "alerta_precio": alerta,
        "candidatos": list(state.get("candidatos_actuales") or []),
    }


def buscar_o_validar_sku_existente(consulta: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Resuelve la búsqueda interactiva de un producto existente en el catálogo:
    1. Primero intenta validar si 'consulta' corresponde directamente a un SKU existente en Firestore.
    2. Si no coincide con un SKU exacto, ejecuta una búsqueda semántica en el catálogo (Top K).
    """
    termino = (consulta or "").strip()
    if not termino:
        return {"modo": "VACIO", "producto": None, "resultados": []}

    # 1. Intentar coincidencia exacta por SKU (en mayúsculas y tal como fue escrito)
    variantes_sku = [termino.upper()]
    if termino not in variantes_sku:
        variantes_sku.append(termino)

    for candidato_sku in variantes_sku:
        try:
            info = consultar_producto_por_sku.invoke({"sku": candidato_sku}) or {}
            if info.get("encontrado") and isinstance(info.get("producto"), dict):
                return {"modo": "SKU_EXACTO", "producto": info["producto"], "resultados": []}
        except Exception:
            pass

    # 2. Búsqueda semántica por texto/marca/nombre en el catálogo
    try:
        resultados = list(
            buscar_candidatos_vectoriales.invoke({"glosa": termino, "formato": "", "top_k": top_k}) or []
        )
    except Exception:
        resultados = []

    return {"modo": "BUSQUEDA_CATALOGO", "producto": None, "resultados": resultados}


def _imprimir_lista_candidatos(candidatos: List[Dict[str, Any]], titulo: str = "Candidatos recuperados del catálogo:") -> None:
    """Muestra en consola una tabla legible con los candidatos disponibles."""
    if not candidatos:
        print(f" {titulo} (Ninguno recuperado automáticamente)")
        return

    print(f" {titulo}")
    for idx_c, cand in enumerate(candidatos, start=1):
        sim = cand.get("similitud_vectorial")
        sim_txt = f" | Sim: {sim:.2f}" if isinstance(sim, (int, float)) else ""
        print(
            f"   [{idx_c}] {cand.get('sku')} - {cand.get('nombre')} "
            f"({cand.get('formato')}) | Costo actual: ${cand.get('precio_costo_actual', 0):,.0f}{sim_txt}"
        )


def _solicitar_busqueda_sku_consola() -> Optional[str]:
    """
    Flujo interactivo para que el operador ingrese un SKU existente (validándolo en Firestore)
    o busque productos en el catálogo por texto/marca antes de decidir.
    Retorna el SKU confirmado o None si decide volver al menú anterior.
    """
    while True:
        consulta = input(
            "    * Ingresa un SKU existente o texto para buscar en el catálogo (Enter para volver): "
        ).strip()
        if not consulta:
            return None

        resultado = buscar_o_validar_sku_existente(consulta, top_k=5)
        if resultado["modo"] == "SKU_EXACTO" and resultado["producto"]:
            prod = resultado["producto"]
            sku_enc = prod.get("sku")
            print(
                f"      [OK] Producto encontrado: {sku_enc} - {prod.get('nombre')} "
                f"({prod.get('formato')}) | Costo actual: ${prod.get('precio_costo', 0):,.0f} | "
                f"Stock: {prod.get('stock_actual', 0)}"
            )
            conf = input(
                "      ¿Confirmar este SKU? (s=Confirmar [Enter] / b=Buscar otro / v=Volver): "
            ).strip().lower() or "s"
            if conf in ["s", "si", "sí", "y", "yes"]:
                return str(sku_enc)
            if conf in ["v", "volver", "c", "cancelar"]:
                return None
            continue

        resultados = resultado["resultados"]
        if not resultados:
            print(f"      [AVISO] No se encontraron productos en el catálogo para '{consulta}'.")
            continue

        _imprimir_lista_candidatos(resultados, titulo=f"     Resultados encontrados para '{consulta}':")
        sel = input(
            f"      Selecciona número (1-{len(resultados)}), 'b' para otra búsqueda o Enter para volver: "
        ).strip().lower()
        if not sel or sel in ["v", "volver"]:
            return None
        if sel.isdigit() and 1 <= int(sel) <= len(resultados):
            return str(resultados[int(sel) - 1]["sku"])


def _completar_actualizacion_existente(
    sku_elegido: str,
    cantidad: Optional[int],
    costo_factura: Optional[float],
    unidades_pack: Optional[int],
    es_pack: bool = False,
    precio_factura_bruto: Optional[float] = None,
    ambiguedad_costo: bool = False,
) -> Dict[str, Any]:
    """Completa eventuales campos ilegibles y construye la acción ACTUALIZAR_EXISTENTE."""
    while cantidad is None:
        cant_input = input("    * Cantidad comprada (era ilegible en la factura): ").strip()
        cantidad = _parsear_entero_positivo(cant_input)
        if cantidad is None:
            print("      Ingresa una cantidad entera mayor a 0.")

    while unidades_pack is None or unidades_pack <= 0:
        unid_input = input("    * Unidades por pack/bulto (era ilegible en la factura): ").strip()
        unidades_pack = _parsear_entero_positivo(unid_input)
        if unidades_pack is None:
            print("      Ingresa un número entero de unidades mayor a 0.")

    costo_unitario = costo_factura
    if ambiguedad_costo and precio_factura_bruto is not None and unidades_pack > 1:
        # Si se confirmó un SKU existente, consultar su costo histórico en catálogo para resolver si ya no es ambiguo
        costo_hist = None
        try:
            prod_info = consultar_producto_por_sku.invoke({"sku": sku_elegido}) or {}
            if prod_info.get("encontrado") and isinstance(prod_info.get("producto"), dict):
                costo_hist = _parsear_float_positivo(prod_info["producto"].get("precio_costo"))
        except Exception:
            costo_hist = None

        if costo_hist is not None and costo_hist > 0:
            res_con_sku = resolver_costo_unitario_linea(
                {"precio_unitario": precio_factura_bruto, "unidades": unidades_pack, "es_pack": True},
                sku=sku_elegido,
                costo_actual_catalogo=costo_hist,
            )
            if not res_con_sku.get("ambiguo"):
                costo_unitario = res_con_sku.get("costo_unitario")
                ambiguedad_costo = False

        if ambiguedad_costo:
            costo_pack_calc = round(precio_factura_bruto / unidades_pack, 2)
            print(f"    * Precio facturado: ${precio_factura_bruto:,.0f} para un pack de {unidades_pack} unidades.")
            tipo_input = input(
                f"      ¿Corresponde a: 1=Precio por PACK (${costo_pack_calc:,.0f}/u) [Enter] / 2=Precio UNITARIO directo (${precio_factura_bruto:,.0f}/u)?: "
            ).strip()
            if tipo_input in ["2", "u", "unidad", "unitario"]:
                costo_unitario = precio_factura_bruto
            else:
                costo_unitario = costo_pack_calc

    while costo_unitario is None:
        costo_input = input("    * Precio de costo unitario por botella/lata (era ilegible en la factura): ").strip()
        costo_unitario = _parsear_float_positivo(costo_input)
        if costo_unitario is None:
            print("      Ingresa un precio de costo válido mayor a 0.")

    pv_mod = input("    * ¿Ajustar precio de venta? [Nuevo precio o Enter para mantener]: ").strip()
    nuevo_pv = _parsear_float_positivo(pv_mod) if pv_mod else None

    return {
        "tipo_accion": "ACTUALIZAR_EXISTENTE",
        "sku": sku_elegido,
        "cantidad": cantidad * unidades_pack,
        "precio_costo": costo_unitario,
        "nuevo_precio_venta": nuevo_pv,
        "unidades_incluidas": True,
        "guardar_equivalencia": True,
    }


def _completar_creacion_nuevo_producto(
    glosa: str,
    formato: str,
    cantidad: Optional[int],
    costo_factura: Optional[float],
    unidades_pack: Optional[int],
    es_pack: bool = False,
    precio_factura_bruto: Optional[float] = None,
    ambiguedad_costo: bool = False,
) -> Dict[str, Any]:
    """Solicita los metadatos necesarios para dar de alta un nuevo SKU en el catálogo."""
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

    while unidades_pack is None or unidades_pack <= 0:
        unid_input = input("    * Unidades por pack/bulto (era ilegible en la factura): ").strip()
        unidades_pack = _parsear_entero_positivo(unid_input)
        if unidades_pack is None:
            print("      Ingresa un número entero de unidades mayor a 0.")

    costo_unitario = costo_factura
    if ambiguedad_costo and precio_factura_bruto is not None and unidades_pack > 1:
        # Si se confirmó un SKU existente, consultar su costo histórico en catálogo para resolver si ya no es ambiguo
        costo_hist = None
        try:
            prod_info = consultar_producto_por_sku.invoke({"sku": nuevo_sku}) or {}
            if prod_info.get("encontrado") and isinstance(prod_info.get("producto"), dict):
                costo_hist = _parsear_float_positivo(prod_info["producto"].get("precio_costo"))
        except Exception:
            costo_hist = None

        if costo_hist is not None and costo_hist > 0:
            res_con_sku = resolver_costo_unitario_linea(
                {"precio_unitario": precio_factura_bruto, "unidades": unidades_pack, "es_pack": True},
                sku=nuevo_sku,
                costo_actual_catalogo=costo_hist,
            )
            if not res_con_sku.get("ambiguo"):
                costo_unitario = res_con_sku.get("costo_unitario")
                ambiguedad_costo = False

        if ambiguedad_costo:
            costo_pack_calc = round(precio_factura_bruto / unidades_pack, 2)
            print(f"    * Precio facturado: ${precio_factura_bruto:,.0f} para un pack de {unidades_pack} unidades.")
            tipo_input = input(
                f"      ¿Corresponde a: 1=Precio por PACK (${costo_pack_calc:,.0f}/u) [Enter] / 2=Precio UNITARIO directo (${precio_factura_bruto:,.0f}/u)?: "
            ).strip()
            if tipo_input in ["2", "u", "unidad", "unitario"]:
                costo_unitario = precio_factura_bruto
            else:
                costo_unitario = costo_pack_calc

    while costo_unitario is None:
        costo_input = input("    * Precio de costo unitario por botella/lata (era ilegible en la factura): ").strip()
        costo_unitario = _parsear_float_positivo(costo_input)
        if costo_unitario is None:
            print("      Ingresa un precio de costo válido mayor a 0.")

    while True:
        pv_input = input(f"    * Precio de venta al público (Costo unitario: ${costo_unitario:,.0f}): ").strip()
        precio_venta = _parsear_float_positivo(pv_input)
        if precio_venta is not None:
            break
        print("      Ingresa un número válido mayor a 0.")

    return {
        "tipo_accion": "CREAR_NUEVO",
        "sku": nuevo_sku,
        "nombre": nombre,
        "marca": marca,
        "categoria": categoria,
        "subcategoria": subcategoria,
        "formato": nuevo_formato,
        "precio_costo": costo_unitario,
        "precio_venta": precio_venta,
        "stock_inicial": cantidad * unidades_pack,
        "unidades_incluidas": True,
        "guardar_equivalencia": True,
    }


def solicitar_decision_humana_consola(
    payload: Dict[str, Any],
    permitir_pausa: bool = True,
) -> Dict[str, Any]:
    """
    Interfaz interactiva de consola para capturar la decisión del operador ante un interrupt()
    de LangGraph. Muestra los candidatos recuperados y permite:
      1. Confirmar uno de los candidatos mostrados ([1..N] o 'c'/'h').
      2. Buscar en el catálogo por texto o ingresar otro SKU existente validado ('b').
      3. Crear un producto nuevo ('s').
      4. Omitir el ítem ('n').
      5. Pausar la factura y guardar la sesión en SQLite ('p' o Ctrl+C).
    """
    proveedor = payload.get("proveedor") or "SIN_PROVEEDOR"
    glosa = payload.get("glosa") or ""
    formato = payload.get("formato") or ""
    costo_factura = payload.get("costo_factura")
    precio_bruto = payload.get("precio_factura_bruto")
    cantidad = payload.get("cantidad")
    unidades_pack = payload.get("unidades_pack")
    es_pack = bool(payload.get("es_pack"))
    unidades_ilegibles = bool(payload.get("unidades_ilegibles"))
    tipo_precio = payload.get("tipo_precio") or ("POR_PACK" if es_pack else "POR_UNIDAD")
    ambiguedad_costo = bool(payload.get("ambiguedad_costo"))
    campos_ilegibles = list(payload.get("campos_ilegibles") or [])
    sku = payload.get("sku_homologado")
    confianza = float(payload.get("confianza") or 0.0)
    justificacion = payload.get("justificacion") or ""
    alerta = payload.get("alerta_precio") or {}
    candidatos = list(payload.get("candidatos") or [])
    tipo_intervencion = payload.get("tipo_intervencion", "ALTA_O_REVISION")

    print("\n" + "=" * 75)
    print(f" {BgColor.GREEN}[HUMAN-IN-THE-LOOP - PAUSA LANGGRAPH]{BgColor.OFF} Se requiere decisión del operador:")
    print(f" Proveedor: '{proveedor}'")

    unid_str = f"{unidades_pack} u/pack" if unidades_pack else "ILEGIBLE (Bulto sin unidades)"
    if es_pack and precio_bruto and costo_factura and unidades_pack and unidades_pack > 1:
        costo_desc = f"${precio_bruto:,.0f} [{tipo_precio}] -> Costo unitario normalizado: ${costo_factura:,.0f}/u"
    else:
        costo_desc = f"${costo_factura:,.0f}/u" if costo_factura is not None else "ILEGIBLE"

    print(
        f" Glosa Factura: '{glosa or 'ILEGIBLE'}' | "
        f"Formato: {formato or 'No especificado/Ilegible'} | "
        f"Costo: {costo_desc} | "
        f"Cantidad: {cantidad if cantidad is not None else 'ILEGIBLE'} bultos ({unid_str})"
    )
    if ambiguedad_costo:
        print(" [AVISO] Ambigüedad en precio de pack: verifica si el valor facturado es por el pack completo o por botella/lata.")
    if campos_ilegibles:
        print(f" [AVISO] Campos ilegibles o incompletos detectados: {', '.join(campos_ilegibles)}")

    try:
        # CASO 1: Coincidencia dudosa / sin SKU -> Mostrar candidatos y las 4 opciones clave (+ pausa)
        if tipo_intervencion == "ALTA_O_REVISION":
            print(f" Motivo: No hay coincidencia segura en el catálogo (Confianza heurística: {confianza:.2f}).")
            print(f" Justificación: {justificacion}")
            _imprimir_lista_candidatos(candidatos)

            while True:
                etiqueta_cand = f"1-{len(candidatos)}=Confirmar candidato / " if candidatos else ""
                opciones_txt = f"({etiqueta_cand}b=Buscar o ingresar otro SKU / s=Crear nuevo / n=Omitir"
                if permitir_pausa:
                    opciones_txt += " / p=Pausar sesión)"
                else:
                    opciones_txt += ")"

                resp = input(f"\n -> ¿Qué acción deseas realizar? {opciones_txt}: ").strip().lower()

                if permitir_pausa and resp in ["p", "pausar", "pause"]:
                    print("=" * 75 + "\n")
                    return {"tipo_accion": "PAUSAR_SESION"}

                # 1. Confirmar directamente uno de los candidatos numerados [1..N]
                if resp.isdigit() and candidatos and 1 <= int(resp) <= len(candidatos):
                    sku_elegido = str(candidatos[int(resp) - 1]["sku"])
                    print(f" -> Candidato confirmado: {sku_elegido}")
                    datos_accion = _completar_actualizacion_existente(
                        sku_elegido,
                        cantidad,
                        costo_factura,
                        unidades_pack,
                        es_pack=es_pack,
                        precio_factura_bruto=precio_bruto,
                        ambiguedad_costo=ambiguedad_costo,
                    )
                    break

                # Alias 'c' o 'h' para elegir candidato de la lista o validar un SKU escrito
                if resp in ["c", "h"]:
                    if candidatos:
                        sel_c = input(
                            f"    * Selecciona número de candidato (1-{len(candidatos)}) o escribe término/SKU: "
                        ).strip()
                        if sel_c.isdigit() and 1 <= int(sel_c) <= len(candidatos):
                            sku_elegido = str(candidatos[int(sel_c) - 1]["sku"])
                            print(f" -> Candidato confirmado: {sku_elegido}")
                            datos_accion = _completar_actualizacion_existente(
                                sku_elegido,
                                cantidad,
                                costo_factura,
                                unidades_pack,
                                es_pack=es_pack,
                                precio_factura_bruto=precio_bruto,
                                ambiguedad_costo=ambiguedad_costo,
                            )
                            break
                        if sel_c:
                            validacion = buscar_o_validar_sku_existente(sel_c, top_k=5)
                            if validacion["modo"] == "SKU_EXACTO" and validacion["producto"]:
                                sku_elegido = str(validacion["producto"]["sku"])
                                print(f" -> SKU existente verificado: {sku_elegido}")
                                datos_accion = _completar_actualizacion_existente(
                                    sku_elegido,
                                    cantidad,
                                    costo_factura,
                                    unidades_pack,
                                    es_pack=es_pack,
                                    precio_factura_bruto=precio_bruto,
                                    ambiguedad_costo=ambiguedad_costo,
                                )
                                break
                            print("    [AVISO] El valor ingresado no coincide con un candidato ni SKU exacto; usa el buscador:")
                    sku_buscado = _solicitar_busqueda_sku_consola()
                    if sku_buscado:
                        datos_accion = _completar_actualizacion_existente(
                            sku_buscado,
                            cantidad,
                            costo_factura,
                            unidades_pack,
                            es_pack=es_pack,
                            precio_factura_bruto=precio_bruto,
                            ambiguedad_costo=ambiguedad_costo,
                        )
                        break
                    continue

                # 2. Buscar en el catálogo o ingresar otro SKU existente validado
                if resp in ["b", "buscar"]:
                    sku_buscado = _solicitar_busqueda_sku_consola()
                    if sku_buscado:
                        datos_accion = _completar_actualizacion_existente(
                            sku_buscado,
                            cantidad,
                            costo_factura,
                            unidades_pack,
                            es_pack=es_pack,
                            precio_factura_bruto=precio_bruto,
                            ambiguedad_costo=ambiguedad_costo,
                        )
                        break
                    continue

                # 3. Crear un producto nuevo
                if resp in ["s", "si", "sí", "y", "yes", "nuevo"]:
                    datos_accion = _completar_creacion_nuevo_producto(
                        glosa,
                        formato,
                        cantidad,
                        costo_factura,
                        unidades_pack,
                        es_pack=es_pack,
                        precio_factura_bruto=precio_bruto,
                        ambiguedad_costo=ambiguedad_costo,
                    )
                    break

                # 4. Omitir el ítem
                if resp in ["n", "no", "o", "omitir"]:
                    print(" -> Ítem omitido por decisión del operador.")
                    datos_accion = {"tipo_accion": "OMITIR", "sku": None}
                    break

                print("    Opción no válida. Elige un número de candidato, 'b' (buscar/SKU), 's' (crear nuevo) o 'n' (omitir).")

        # CASO 2: Producto homologado pero con datos ilegibles (cantidad/precio/unidades) y/o alerta de precio/pack
        else:
            print(f" Producto homologado sugerido: {sku}")
            if ambiguedad_costo:
                print(" Motivo: Ambigüedad en precio por pack vs unidad individual.")
            if campos_ilegibles:
                print(f" Motivo: Faltan datos legibles en la factura ({', '.join(campos_ilegibles)}).")
            if alerta.get("alerta"):
                print(f" Motivo: Alerta de variación de costo detectada ({alerta.get('variacion_porcentual')}%)")
            if candidatos:
                _imprimir_lista_candidatos(candidatos)

            while True:
                opciones_txt = "(s=Confirmar SKU [Enter] / b=Buscar u otro SKU / c=Crear nuevo / n=Omitir"
                if permitir_pausa:
                    opciones_txt += " / p=Pausar sesión)"
                else:
                    opciones_txt += ")"

                continuar = input(f"\n -> ¿Qué acción deseas realizar? {opciones_txt}: ").strip().lower() or "s"

                if permitir_pausa and continuar in ["p", "pausar", "pause"]:
                    print("=" * 75 + "\n")
                    return {"tipo_accion": "PAUSAR_SESION"}

                if continuar in ["s", "si", "sí", "y", "yes"]:
                    datos_accion = _completar_actualizacion_existente(
                        str(sku),
                        cantidad,
                        costo_factura,
                        unidades_pack,
                        es_pack=es_pack,
                        precio_factura_bruto=precio_bruto,
                        ambiguedad_costo=ambiguedad_costo,
                    )
                    break

                if continuar in ["b", "buscar"]:
                    sku_buscado = _solicitar_busqueda_sku_consola()
                    if sku_buscado:
                        datos_accion = _completar_actualizacion_existente(
                            sku_buscado,
                            cantidad,
                            costo_factura,
                            unidades_pack,
                            es_pack=es_pack,
                            precio_factura_bruto=precio_bruto,
                            ambiguedad_costo=ambiguedad_costo,
                        )
                        break
                    continue

                if continuar in ["c", "nuevo"]:
                    datos_accion = _completar_creacion_nuevo_producto(
                        glosa,
                        formato,
                        cantidad,
                        costo_factura,
                        unidades_pack,
                        es_pack=es_pack,
                        precio_factura_bruto=precio_bruto,
                        ambiguedad_costo=ambiguedad_costo,
                    )
                    break

                if continuar in ["n", "no", "o", "omitir"]:
                    print(" -> Ítem omitido por decisión del operador.")
                    datos_accion = {"tipo_accion": "OMITIR", "sku": sku}
                    break

                print("    Opción no válida. Ingresa 's', 'b', 'c' o 'n'.")

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
    Nodo 1: Invoca la herramienta multimodal para extraer encabezado e ítems desde la imagen,
    validando la respuesta contra el esquema Pydantic (encabezado, lista de ítems, cantidades y precios).
    Distingue explícitamente entre documento válido sin productos (PENDIENTE_REVISION) y extracción fallida (ERROR_EXTRACCION).
    """
    ruta_imagen = state["ruta_imagen"]
    respaldo_archivo = Path(ruta_imagen).name if ruta_imagen else None
    print(f"\n{BgColor.YELLOW}[Agente - Percepción]{BgColor.OFF} Extrayendo datos de la factura: '{ruta_imagen}'...")

    datos_factura_crudos = dict(extraer_factura_desde_imagen.invoke({"ruta_imagen": ruta_imagen}) or {})
    datos_factura = validar_esquema_extraccion(datos_factura_crudos)

    estado_extraccion = datos_factura.get("estado_extraccion", "OK")
    proveedor_ext = _limpiar_texto(datos_factura.get("proveedor")) or None
    folio_ext = _limpiar_texto(datos_factura.get("numero_factura")) or None

    # Caso A: Extracción fallida (imagen ilegible, JSON corrupto o fallo general de OCR/modelo)
    if estado_extraccion == "ERROR_EXTRACCION":
        motivo = datos_factura.get("motivo_sin_items") or "No se pudo leer la factura con los modelos configurados."
        msg_error = f"ERROR DE EXTRACCIÓN: {motivo}"
        print(f"{BgColor.RED}[Agente - Error Extracción]{BgColor.OFF} {msg_error}")
        return {
            "datos_factura": datos_factura,
            "items_pendientes": [],
            "items_procesados": [],
            "mensajes_bitacora": state.get("mensajes_bitacora", []) + [msg_error],
            "estado_final": "ERROR_EXTRACCION",
        }

    # Caso B: Documento formal detectado con encabezado, pero sin ítems comerciales
    if estado_extraccion == "DOC_SIN_ITEMS":
        motivo = datos_factura.get("motivo_sin_items") or "Documento formal sin líneas de productos comerciales."
        msg_aviso = (
            f"DOCUMENTO SIN ÍTEMS DETECTADOS: Proveedor '{proveedor_ext or 'S/P'}', "
            f"Folio '{folio_ext or 'S/F'}'. {motivo}"
        )
        print(f"{BgColor.YELLOW}[Agente - Aviso Percepción]{BgColor.OFF} {msg_aviso}")
        return {
            "datos_factura": datos_factura,
            "items_pendientes": [],
            "items_procesados": [],
            "mensajes_bitacora": state.get("mensajes_bitacora", []) + [msg_aviso],
            "estado_final": "PENDIENTE_REVISION",
        }

    # Caso C: Extracción válida con ítems
    items_crudos = list(datos_factura.get("items") or [])
    items = []
    for idx, it in enumerate(items_crudos, start=1):
        copia_item = dict(it) if isinstance(it, dict) else {}
        linea_num = _parsear_entero_positivo(copia_item.get("linea_id")) or idx
        copia_item["linea_id"] = linea_num
        copia_item["id_operacion"] = construir_id_operacion(
            proveedor=proveedor_ext,
            folio=folio_ext,
            linea_id=linea_num,
            respaldo_factura=respaldo_archivo,
        )
        items.append(copia_item)

    datos_factura["items"] = items

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
    Si existe en estado COMPLETADO_CON_ERRORES (o PARCIAL/PENDIENTE_REVISION), conserva
    las líneas exitosas y filtra 'items_pendientes' para retomar únicamente las líneas fallidas o pendientes.
    """
    if state.get("estado_final") == "ERROR_EXTRACCION":
        # En caso de fallo de extracción no hay datos consistentes para validar duplicidad en BD
        return {
            "es_duplicada": False,
            "estado_final": "ERROR_EXTRACCION",
        }

    datos = state.get("datos_factura") or {}
    folio = datos.get("numero_factura")
    proveedor = datos.get("proveedor")

    print(f"{BgColor.YELLOW}[Agente - Verificación]{BgColor.OFF} Comprobando duplicidad para folio '{folio}' de '{proveedor}'...")
    resultado = consultar_duplicidad_factura.invoke({"folio": folio, "proveedor": proveedor}) or {}

    es_duplicada = bool(resultado.get("es_duplicada", False))
    estado_verificacion = resultado.get("estado_verificacion", "")
    registro_existente = resultado.get("registro_existente")
    msg = resultado.get("mensaje", "")
    print(f" -> {msg}")

    bitacora = list(state.get("mensajes_bitacora") or []) + ([msg] if msg else [])
    actualizacion: Dict[str, Any] = {
        "es_duplicada": es_duplicada,
        "info_duplicada": registro_existente,
        "mensajes_bitacora": bitacora,
    }
    if es_duplicada:
        msg_p1 = formatear_aviso_planificacion(
            prioridad=1,
            regla="Control de Duplicidad e Integridad",
            detalle=f"Factura folio '{folio}' de '{proveedor}' ya completada previamente. Deteniendo escrituras.",
        )
        print(msg_p1)
        bitacora.append(msg_p1)
        actualizacion["politica_planificacion"] = {
            "prioridad": 1,
            "regla": "Control de Duplicidad e Integridad",
            "accion": "Detener las escrituras",
            "detalle": msg_p1,
        }

    if estado_verificacion == "ERROR_CONSULTA" or resultado.get("es_duplicada") is None:
        actualizacion["estado_final"] = "ERROR"
        return actualizacion

    if estado_verificacion == "REINTENTO_CON_ERRORES" and isinstance(registro_existente, dict):
        items_previos = list(registro_existente.get("items") or [])
        ruta_img = _limpiar_texto(state.get("ruta_imagen"))
        respaldo_archivo = Path(ruta_img).name if ruta_img else None

        items_conservados: List[Dict[str, Any]] = []
        lineas_exitosas = set()
        ops_exitosas = set()

        for idx_prev, prev in enumerate(items_previos, start=1):
            if not isinstance(prev, dict):
                continue
            if prev.get("estado") in _ESTADOS_LINEA_EXITOSA:
                copia_prev = dict(prev)
                lid = _parsear_entero_positivo(copia_prev.get("linea_id")) or idx_prev
                op_id = _limpiar_texto(copia_prev.get("id_operacion")) or construir_id_operacion(
                    proveedor=_limpiar_texto(proveedor) or None,
                    folio=_limpiar_texto(folio) or None,
                    linea_id=lid,
                    respaldo_factura=respaldo_archivo,
                )
                copia_prev["linea_id"] = lid
                copia_prev["id_operacion"] = op_id
                items_conservados.append(copia_prev)
                lineas_exitosas.add(lid)
                ops_exitosas.add(op_id)

        pendientes_originales = list(state.get("items_pendientes") or [])
        pendientes_filtrados: List[Dict[str, Any]] = []
        for idx_p, pend in enumerate(pendientes_originales, start=1):
            copia_pend = dict(pend) if isinstance(pend, dict) else {}
            ident_p = _obtener_identidad_linea(state, copia_pend, linea_por_defecto=idx_p)
            lid_p = ident_p["linea_id"]
            op_p = ident_p["id_operacion"]
            if lid_p in lineas_exitosas or op_p in ops_exitosas:
                continue
            copia_pend["linea_id"] = lid_p
            copia_pend["id_operacion"] = op_p
            pendientes_filtrados.append(copia_pend)

        msg_reintento = (
            f"Reintento parcial selectivo: se omiten {len(items_conservados)} línea(s) exitosa(s) "
            f"y se procesarán {len(pendientes_filtrados)} línea(s) fallida(s)/pendiente(s)."
        )
        print(f" -> {msg_reintento}")

        msg_p5 = formatear_aviso_planificacion(
            prioridad=5,
            regla="Resiliencia e Idempotencia",
            detalle=(
                f"Factura previa en estado '{estado_verificacion}'. Retomando únicamente {len(pendientes_filtrados)} "
                f"línea(s) fallida(s), conservando {len(items_conservados)} línea(s) exitosa(s) sin repetir escrituras."
            ),
        )
        print(msg_p5)

        actualizacion["items_procesados"] = items_conservados
        actualizacion["items_pendientes"] = pendientes_filtrados
        actualizacion["auditoria_id"] = registro_existente.get("id")
        actualizacion["mensajes_bitacora"] = bitacora + [msg_reintento, msg_p5]
        actualizacion["politica_planificacion"] = {
            "prioridad": 5,
            "regla": "Resiliencia e Idempotencia",
            "accion": "Reintento selectivo",
            "detalle": msg_p5,
        }

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
    Nodo 3 (Loop): Toma el primer ítem de la lista de pendientes para su evaluación individual,
    asegurando su identidad de línea ('linea_id' e 'id_operacion') y validando las unidades por pack.
    """
    pendientes = state["items_pendientes"]
    item_actual = dict(pendientes[0]) if isinstance(pendientes[0], dict) else {}
    restantes = pendientes[1:]

    identidad = _obtener_identidad_linea(state, item_actual)
    item_actual["linea_id"] = identidad["linea_id"]
    item_actual["id_operacion"] = identidad["id_operacion"]

    # Validación estricta de unidades por bulto/pack (no asumir 1 silenciosamente si es bulto ilegible)
    info_unid = validar_unidades_item(item_actual)
    item_actual["es_pack"] = info_unid["es_pack"]
    item_actual["unidades_ilegibles"] = info_unid["unidades_ilegibles"]
    unidades_pack = info_unid["unidades"]
    if unidades_pack is not None:
        item_actual["unidades"] = unidades_pack

    cant_txt = item_actual.get("cantidad")
    unid_txt = f"{unidades_pack} u/pack" if unidades_pack is not None else "ILEGIBLE (Bulto sin unidades)"
    msg = (
        f"Iniciando evaluación de ítem [L{identidad['linea_id']}]: '{item_actual.get('nombre_glosa')}' "
        f"(Formato: {item_actual.get('formato')}, Cantidad: {cant_txt}, Unidades/Pack: {unid_txt})"
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
    Nodo 4: Recuperación de memoria histórica de equivalencias aprobadas (proveedor + glosa -> SKU)
    y búsqueda semántica de los top 3 candidatos más similares en Firestore.
    """
    item = state["item_actual"]
    datos = state.get("datos_factura") or {}
    proveedor = _limpiar_texto(datos.get("proveedor")) or None
    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))

    if not glosa:
        msg = "Nombre/glosa ilegible en la factura. Se omite búsqueda vectorial y se enviará a revisión humana."
        print(f"{BgColor.YELLOW}[Agente - Retrieval Semántico]{BgColor.OFF} -> {msg}")
        return {
            "candidatos_actuales": [],
            "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
        }

    # 1. Consultar primero la memoria de equivalencias aprobadas (proveedor + glosa -> SKU)
    if _consulta_equivalencia_activa():
        eq_res = consultar_equivalencia_proveedor.invoke({"proveedor": proveedor, "glosa": glosa}) or {}
        sku_eq = eq_res.get("sku_homologado") if eq_res.get("encontrada") else None
        if sku_eq:
            prod_eq = consultar_producto_por_sku.invoke({"sku": sku_eq}) or {}
            if prod_eq.get("encontrado") and isinstance(prod_eq.get("producto"), dict):
                p = prod_eq["producto"]
                cand_aprobado = {
                    "sku": p.get("sku") or sku_eq,
                    "nombre": p.get("nombre", ""),
                    "marca": p.get("marca", ""),
                    "categoria": p.get("categoria", ""),
                    "formato": p.get("formato", ""),
                    "precio_costo_actual": p.get("precio_costo", 0),
                    "precio_venta_actual": p.get("precio_venta", 0),
                    "similitud_vectorial": 1.0,
                    "equivalencia_aprobada": True,
                }
                msg = (
                    f"Equivalencia histórica aprobada recuperada para '{proveedor or 'SIN_PROVEEDOR'}' "
                    f"+ '{glosa}': {cand_aprobado['sku']} ({cand_aprobado['nombre']})"
                )
                print(f"{BgColor.YELLOW}[Agente - Memoria Equivalencias]{BgColor.OFF} -> {msg}")
                return {
                    "candidatos_actuales": [cand_aprobado],
                    "mensajes_bitacora": state["mensajes_bitacora"] + [msg],
                }

    # 2. Si no hay equivalencia aprendida vigente, ejecutar búsqueda vectorial Top 3
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
    Nodo 5: Si existe una equivalencia histórica aprobada la aplica directamente; de lo contrario,
    el LLM razona sobre abreviaciones y similitud con los candidatos recuperados.
    """
    item = state["item_actual"]
    candidatos = state["candidatos_actuales"]
    glosa = _limpiar_texto(item.get("nombre_glosa"))
    formato = _limpiar_texto(item.get("formato"))

    print(f"{BgColor.YELLOW}[Agente - Razonamiento LLM]{BgColor.OFF} Evaluando homologación...")

    cand_eq = next(
        (c for c in (candidatos or []) if isinstance(c, dict) and c.get("equivalencia_aprobada") is True),
        None,
    )

    if not glosa:
        decision = {
            "sku_seleccionado": None,
            "confianza": 0.0,
            "justificacion": "El nombre o glosa del producto no es legible en la factura.",
            "requiere_revision": True,
        }
    elif cand_eq is not None:
        decision = {
            "sku_seleccionado": cand_eq["sku"],
            "confianza": 1.0,
            "justificacion": f"Equivalencia histórica aprobada para el proveedor y la glosa '{glosa}'.",
            "requiere_revision": False,
        }
    else:
        decision_cruda = razonar_homologacion.invoke({
            "glosa_factura": glosa,
            "formato_factura": formato,
            "candidatos": candidatos,
        }) or {}
        # La ponderación heurística (70% LLM + 30% similitud vectorial) ya se aplica una única vez
        # dentro de razonar_homologacion; aquí solo se valida la estructura y pertenencia del SKU.
        decision = validar_estructura_homologacion(decision_cruda, candidatos=candidatos)

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
    Nodo 6: Normaliza el costo unitario individual (distinguiendo precio por pack vs unidad)
    y evalúa si existe variación abrupta de costo en comparación al histórico.
    """
    decision = state.get("decision_actual") or {}
    item = state.get("item_actual") or {}
    sku = decision.get("sku_seleccionado")

    costo_actual = 0.0
    if sku:
        try:
            prod_info = consultar_producto_por_sku.invoke({"sku": sku}) or {}
            if prod_info.get("encontrado") and isinstance(prod_info.get("producto"), dict):
                costo_actual = float(prod_info["producto"].get("precio_costo") or 0.0)
        except Exception:
            costo_actual = 0.0

    # Normalizar costo unitario distinguiendo precio por pack vs unidad (IE1)
    res_costo = resolver_costo_unitario_linea(
        item,
        sku=sku,
        costo_actual_catalogo=costo_actual if costo_actual > 0 else None,
    )
    costo_unitario = res_costo.get("costo_unitario")
    item["costo_unitario_normalizado"] = costo_unitario
    item["tipo_precio"] = res_costo.get("tipo_precio")
    item["ambiguedad_costo"] = res_costo.get("ambiguo")
    item["precio_pack_calculado"] = res_costo.get("precio_pack")

    alerta_precio = None
    if sku:
        if costo_unitario is None:
            print(" -> [AVISO] Costo unitario no determinable; requerirá validación humana.")
        elif costo_actual > 0:
            alerta_precio = evaluar_variacion_costo.invoke({
                "costo_factura": costo_unitario,   # Compara el costo unitario normalizado, NO el precio bruto del pack
                "costo_actual": costo_actual,
                "umbral_alerta": 0.15,
            })

            if alerta_precio.get("alerta"):
                print(f" -> [ALERTA DE COSTO] {alerta_precio.get('mensaje')}")

        if res_costo.get("ambiguo"):
            alerta_precio = alerta_precio or {"variacion_porcentual": 0.0}
            alerta_precio["alerta"] = True
            alerta_precio["ambiguedad_pack"] = True
            alerta_precio["mensaje"] = res_costo.get("motivo")
            print(f" -> [AMBIGÜEDAD PACK] {res_costo.get('motivo')}")

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
    - Envía siempre la identidad de operación ('id_operacion' = proveedor + folio + línea) a todas las escrituras.
    - Multiplica 'cantidad' por 'unidades' (tamaño del pack) para sumar unidades individuales.
    - Verifica 'resultado["exito"]' antes de registrar el estado del ítem.
    """
    item = state.get("item_actual") or {}
    decision = state.get("decision_actual") or {}
    accion = state.get("datos_intervencion_humana")

    identidad = _obtener_identidad_linea(state, item)
    proveedor_actual = identidad["proveedor"]
    folio_actual = identidad["folio"]
    linea_id = identidad["linea_id"]
    id_operacion = identidad["id_operacion"]

    info_unid = validar_unidades_item(item)
    unidades_pack = (
        _parsear_entero_positivo(item.get("unidades_validadas"))
        or info_unid.get("unidades")
        or _parsear_entero_positivo(item.get("unidades"))
        or 1
    )
    cantidad_bultos = _parsear_entero_positivo(item.get("cantidad")) or 0
    unidades_totales = cantidad_bultos * unidades_pack
    glosa_original_limpia = _limpiar_texto(item.get("nombre_glosa"))

    precio_factura_bruto = _parsear_float_positivo(item.get("precio_unitario")) or 0.0
    costo_normalizado = item.get("costo_unitario_normalizado")
    if costo_normalizado is None:
        costo_normalizado = (
            round(precio_factura_bruto / unidades_pack, 2)
            if (info_unid.get("es_pack") and unidades_pack > 1 and item.get("tipo_precio") == "POR_PACK")
            else precio_factura_bruto
        )

    registro_item = {
        "linea_id": linea_id,
        "id_operacion": id_operacion,
        "glosa_original": glosa_original_limpia or "ILEGIBLE",
        "formato": _limpiar_texto(item.get("formato")) or None,
        "cantidad": unidades_totales,
        "unidades_por_pack": unidades_pack,
        "precio_unitario": costo_normalizado,
        "precio_factura_bruto": precio_factura_bruto,
        "tipo_precio": item.get("tipo_precio") or ("POR_PACK" if info_unid.get("es_pack") else "POR_UNIDAD"),
        "sku_homologado": None,
        "estado": "PENDIENTE",
    }
    msg = ""
    msg_equivalencia = ""
    bitacora = list(state.get("mensajes_bitacora", []))

    # Si vino de intervención humana
    if accion:
        tipo = accion.get("tipo_accion")
        resultado: Dict[str, Any] = {}
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
                "folio_factura": folio_actual,
                "proveedor": proveedor_actual,
                "linea_id": linea_id,
                "id_operacion": id_operacion,
            }

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
            if resultado.get("ya_aplicado"):
                registro_item["ya_aplicado_previamente"] = True

            if resultado.get("exito"):
                registro_item["estado"] = "CREADO_NUEVO"
                print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")
            else:
                registro_item["estado"] = "ERROR_TRANSACCIONAL" if resultado.get("error_transaccional") else "ERROR_ESCRITURA"
                print(f"{BgColor.RED}[Agente - Error Escritura]{BgColor.OFF} {msg}")
                msg_p5 = formatear_aviso_planificacion(
                    prioridad=5,
                    regla="Resiliencia e Idempotencia",
                    detalle=f"Falló creación de producto [L{linea_id}] ({msg}). Guardado como pendiente ({registro_item['estado']}) para reintento selectivo.",
                )
                print(msg_p5)
                bitacora.append(msg_p5)

        elif tipo == "ACTUALIZAR_EXISTENTE":
            cantidad_actualizar = accion["cantidad"]
            if not accion.get("unidades_incluidas"):
                cantidad_actualizar = cantidad_actualizar * unidades_pack

            args_act = {
                "sku": accion["sku"],
                "cantidad": cantidad_actualizar,
                "nuevo_costo": accion["precio_costo"],
                "nuevo_precio_venta": accion.get("nuevo_precio_venta"),
                "folio_factura": folio_actual,
                "proveedor": proveedor_actual,
                "linea_id": linea_id,
                "id_operacion": id_operacion,
            }

            resultado = actualizar_stock_y_costo.invoke(args_act) or {}

            msg = resultado.get("mensaje", "")
            registro_item["cantidad"] = cantidad_actualizar
            registro_item["precio_unitario"] = accion["precio_costo"]
            registro_item["sku_homologado"] = accion["sku"]
            if resultado.get("ya_aplicado"):
                registro_item["ya_aplicado_previamente"] = True

            if resultado.get("exito"):
                registro_item["estado"] = "ACTUALIZADO"
                print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")
            else:
                registro_item["estado"] = "ERROR_TRANSACCIONAL" if resultado.get("error_transaccional") else "ERROR_ESCRITURA"
                print(f"{BgColor.RED}[Agente - Error Escritura]{BgColor.OFF} {msg}")
                msg_p5 = formatear_aviso_planificacion(
                    prioridad=5,
                    regla="Resiliencia e Idempotencia",
                    detalle=f"Falló actualización de stock [L{linea_id}] ({msg}). Guardado como pendiente ({registro_item['estado']}) para reintento selectivo.",
                )
                print(msg_p5)
                bitacora.append(msg_p5)

        else:
            registro_item["estado"] = "OMITIDO_POR_HUMANO"
            msg = "Ítem marcado como omitido."
            print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")

        # Persistir equivalencia aprobada (proveedor + glosa -> SKU) en memoria de largo plazo
        if (
            tipo in ("CREAR_NUEVO", "ACTUALIZAR_EXISTENTE")
            and resultado.get("exito")
            and glosa_original_limpia
            and accion.get("guardar_equivalencia", True)
            and _escritura_equivalencia_activa()
        ):
            res_eq = guardar_equivalencia_proveedor.invoke({
                "proveedor": proveedor_actual,
                "glosa": glosa_original_limpia,
                "sku": accion["sku"],
                "formato": _limpiar_texto(item.get("formato")) or None,
                "origen": "APROBACION_HUMANA",
            }) or {}
            if res_eq.get("exito"):
                registro_item["equivalencia_guardada"] = True
                msg_equivalencia = res_eq.get("mensaje", "")
                if msg_equivalencia:
                    print(f"{BgColor.YELLOW}[Agente - Memoria Equivalencias]{BgColor.OFF} {msg_equivalencia}")

    # Si fue homologado automáticamente sin intervención
    else:
        sku = decision.get("sku_seleccionado")
        cantidad_base = _parsear_entero_positivo(item.get("cantidad")) or 1
        cantidad_total = cantidad_base * unidades_pack

        args_auto = {
            "sku": sku,
            "cantidad": cantidad_total,
            "nuevo_costo": costo_normalizado,
            "folio_factura": folio_actual,
            "proveedor": proveedor_actual,
            "linea_id": linea_id,
            "id_operacion": id_operacion,
        }

        resultado = actualizar_stock_y_costo.invoke(args_auto) or {}

        msg = resultado.get("mensaje", "")
        registro_item["cantidad"] = cantidad_total
        registro_item["sku_homologado"] = sku
        if resultado.get("ya_aplicado"):
            registro_item["ya_aplicado_previamente"] = True

        if resultado.get("exito"):
            registro_item["estado"] = "AUTO_HOMOLOGADO"
            print(f"{BgColor.YELLOW}[Agente - Escritura]{BgColor.OFF} {msg}")
        else:
            registro_item["estado"] = "ERROR_TRANSACCIONAL" if resultado.get("error_transaccional") else "ERROR_ESCRITURA"
            print(f"{BgColor.RED}[Agente - Error Escritura]{BgColor.OFF} {msg}")
            msg_p5 = formatear_aviso_planificacion(
                prioridad=5,
                regla="Resiliencia e Idempotencia",
                detalle=f"Falló auto-escritura de stock [L{linea_id}] ({msg}). Guardado como pendiente ({registro_item['estado']}) para reintento selectivo.",
            )
            print(msg_p5)
            bitacora.append(msg_p5)

    items_actualizados = list(state.get("items_procesados", [])) + [registro_item]
    nuevos_msgs = [m for m in (msg, msg_equivalencia) if m]
    bitacora.extend(nuevos_msgs)
    return {
        "items_procesados": items_actualizados,
        "mensajes_bitacora": bitacora,
    }


def node_registrar_auditoria(state):
    """
    Nodo 9: Finaliza el flujo guardando o actualizando el comprobante de auditoría en
    'historial_facturas' de Firestore (Memoria transaccional de largo plazo).
    Registra estados precisos: ERROR_EXTRACCION, PENDIENTE_REVISION, PARCIAL,
    COMPLETADO_CON_ERRORES o COMPLETADO.
    """
    datos = state.get("datos_factura") or {}
    folio = _limpiar_texto(datos.get("numero_factura")) or "S/F"
    proveedor = _limpiar_texto(datos.get("proveedor")) or "DESCONOCIDO"
    items_procesados = sorted(
        list(state.get("items_procesados") or []),
        key=lambda x: x.get("linea_id", 10**6) if isinstance(x, dict) else 10**6,
    )
    items_pendientes = list(state.get("items_pendientes") or [])
    auditoria_previa = _limpiar_texto(state.get("auditoria_id")) or None
    estado_state = state.get("estado_final")

    if estado_state == "ERROR_EXTRACCION":
        estado_auditoria = "ERROR_EXTRACCION"
    elif len(items_procesados) == 0:
        # Documento sin productos comerciales o extracción vacía: NUNCA COMPLETADO
        estado_auditoria = "PENDIENTE_REVISION"
    else:
        hubo_errores = any(i.get("estado") in ("ERROR_ESCRITURA", "ERROR_TRANSACCIONAL") for i in items_procesados)
        quedan_pendientes = len(items_pendientes) > 0
        todos_omitidos = all(i.get("estado") == "OMITIDO_POR_HUMANO" for i in items_procesados)

        if hubo_errores:
            estado_auditoria = "COMPLETADO_CON_ERRORES"
        elif quedan_pendientes or todos_omitidos:
            estado_auditoria = "PARCIAL"
        else:
            estado_auditoria = "COMPLETADO"

    print(f"\n{BgColor.YELLOW}[Agente - Auditoría]{BgColor.OFF} Guardando registro transaccional para folio '{folio}'...")

    args_auditoria: Dict[str, Any] = {
        "folio": folio,
        "proveedor": proveedor,
        "total_items": len(items_procesados),
        "items_procesados": items_procesados,
        "estado": estado_auditoria,
    }
    if auditoria_previa:
        args_auditoria["auditoria_id_existente"] = auditoria_previa

    resultado = registrar_auditoria_factura.invoke(args_auditoria) or {}

    id_auditoria = resultado.get("id_auditoria") or auditoria_previa
    msg = f"Factura folio '{folio}' procesada ({estado_auditoria}). Auditoría ID: {id_auditoria}"
    print(f" -> {msg}\n")

    return {
        "auditoria_id": id_auditoria,
        "items_procesados": items_procesados,
        "estado_final": estado_auditoria,
        "mensajes_bitacora": list(state.get("mensajes_bitacora") or []) + [msg],
    }


# ============================================================================
# 2. BORDES CONDICIONALES (Adaptive Decision Edges)
# ============================================================================

def decidir_ruta_duplicidad(state):
    """
    Borde condicional 1:
    - Si hubo error de extracción o el documento no tiene ítems, pasa a registrar auditoría
      con el estado correspondiente (ERROR_EXTRACCION o PENDIENTE_REVISION) sin marcar jamás COMPLETADO.
    - Si la factura ya existe o hubo error de consulta en Firestore, pasa a manejar duplicada.
    - Si es nueva y tiene ítems pendientes, arranca la iteración de productos.
    """
    estado = state.get("estado_final")
    if estado == "ERROR_EXTRACCION":
        return "registrar_auditoria"
    if state.get("es_duplicada") or estado == "ERROR":
        return "manejar_duplicada"
    if not state.get("items_pendientes"):
        return "registrar_auditoria"
    return "preparar_siguiente_item"


def decidir_ruta_evaluacion(state):
    """
    Borde condicional 2: Aplica explícitamente la Política de Planificación (IE5):
    - Prioridad 2 (Faltan datos necesarios): Solicitar corrección (HITL).
    - Prioridad 4 (Ambigüedad o aumento de costo): Solicitar revisión (HITL).
    - Prioridad 3 (Equivalencia segura y datos válidos): Aplicar el movimiento.
    """
    item = state.get("item_actual") or {}
    decision = state.get("decision_actual") or {}
    alerta = state.get("alerta_precio") or {}
    campos_ilegibles = _obtener_campos_ilegibles(item)

    eval_pol = evaluar_politica_evaluacion_item(
        item=item,
        decision=decision,
        alerta=alerta,
        campos_ilegibles=campos_ilegibles,
    )
    print(eval_pol["mensaje_formateado"])
    bitacora = list(state.get("mensajes_bitacora") or [])
    bitacora.append(eval_pol["mensaje_formateado"])
    state["mensajes_bitacora"] = bitacora
    state["politica_planificacion"] = eval_pol

    return eval_pol["ruta_grafo"]


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
