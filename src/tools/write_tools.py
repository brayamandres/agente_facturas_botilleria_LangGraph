"""
Herramientas de Escritura (Writing Tools) para el agente.
Permiten persistir cambios en Firestore con protección transaccional contra escrituras repetidas:
- Identificación única por movimiento: proveedor + folio + identificador de línea.
- Registro del movimiento ('movimientos_inventario') y actualización de stock en una misma transacción.
- Creación/actualización de auditoría en 'historial_facturas' soportando reintentos parciales.
"""
import hashlib
import re
import time
from typing import Any, Optional

from langchain_core.tools import tool
from firebase_admin import firestore
from google.cloud.firestore_v1.base_client import BaseClient
from google.cloud.firestore_v1.transaction import Transaction as FirestoreTransaction
from google.cloud.firestore_v1.vector import Vector

import src.services.firebase as firebase_svc
from src.services.firebase import (
    get_productos_collection,
    get_historial_collection,
    get_movimientos_collection,
    get_equivalencias_collection,
)
from src.services.embeddings import generar_embedding


_VALORES_VACIOS = {"none", "null", "n/a", "s/f", "desconocido", ""}


def _limpiar_campo(valor: Any) -> Optional[str]:
    """Limpia un valor textual descartando marcadores nulos o vacíos."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto.lower() in _VALORES_VACIOS:
        return None
    return texto


def _normalizar_segmento_id(valor: Any, por_defecto: str) -> str:
    """
    Normaliza un segmento para que sea determinístico y seguro como ID de documento en Firestore
    (sin barras '/' ni caracteres de ruta reservados).
    """
    limpio = _limpiar_campo(valor)
    if not limpio:
        return por_defecto
    texto = re.sub(r"[/\\]+", "-", limpio.upper())
    texto = re.sub(r"\s+", "_", texto)
    texto = re.sub(r"[^A-Z0-9_\-\.]", "", texto)
    texto = texto.strip(".")
    return texto[:80] or por_defecto


def construir_id_operacion(
    proveedor: Any = None,
    folio: Any = None,
    linea_id: Any = None,
    respaldo_factura: Any = None,
) -> str:
    """
    Construye la identidad canónica de una operación de escritura por línea de factura:
    '<PROVEEDOR>::<FOLIO>::L<LINEA_ID>'

    Esto permite:
    - Aplicar múltiples líneas del mismo SKU dentro de una misma factura (L1 vs L2).
    - Evitar colisiones cuando distintos proveedores usan el mismo número de folio.
    - Omitir de forma idempotente únicamente las líneas ya aplicadas al reintentar.
    """
    prov_norm = _normalizar_segmento_id(proveedor, "SIN_PROVEEDOR")
    folio_defecto = (
        f"SIN_FOLIO_{_normalizar_segmento_id(respaldo_factura, 'LOCAL')}"
        if _limpiar_campo(respaldo_factura)
        else "SIN_FOLIO"
    )
    folio_norm = _normalizar_segmento_id(folio, folio_defecto)

    linea_txt = _limpiar_campo(linea_id) or "1"
    if linea_txt.upper().startswith("L") and len(linea_txt) > 1:
        linea_txt = linea_txt[1:]
    linea_norm = _normalizar_segmento_id(linea_txt, "1")

    return f"{prov_norm}::{folio_norm}::L{linea_norm}"


def _resolver_id_operacion(
    id_operacion: Any = None,
    proveedor: Any = None,
    folio_factura: Any = None,
    linea_id: Any = None,
) -> Optional[str]:
    """
    Resuelve el identificador de operación efectivo recibido por una herramienta de escritura.
    """
    op_explicito = _limpiar_campo(id_operacion)
    if op_explicito:
        return re.sub(r"[/\\]+", "-", op_explicito)

    if any(_limpiar_campo(v) is not None for v in (proveedor, folio_factura, linea_id)):
        return construir_id_operacion(
            proveedor=proveedor,
            folio=folio_factura,
            linea_id=linea_id,
        )
    return None


def _obtener_coleccion_movimientos(productos_ref: Any) -> Any:
    """
    Obtiene la colección 'movimientos_inventario' asociada al mismo cliente Firestore.
    En entornos de pruebas unitarias donde solo se mockea 'get_productos_collection' sin
    credenciales reales de Firebase, degrada limpiamente a None sin lanzar excepción.
    """
    cliente_db = getattr(productos_ref, "_client", None)
    if isinstance(cliente_db, BaseClient):
        return cliente_db.collection("movimientos_inventario")

    # Si get_movimientos_collection fue mockeado en pruebas o el cliente Singleton ya está activo
    if (
        get_movimientos_collection is not firebase_svc.get_movimientos_collection
        or getattr(firebase_svc, "_firestore_client", None) is not None
    ):
        try:
            return get_movimientos_collection()
        except Exception:
            return None
    return None


def _snapshot_existe(snapshot: Any) -> bool:
    """Comprueba de forma segura si un DocumentSnapshot existe (compatible con mocks)."""
    if snapshot is None:
        return False
    prop = getattr(snapshot, "exists", False)
    return isinstance(prop, bool) and prop is True


@tool
def actualizar_stock_y_costo(
    sku,
    cantidad,
    nuevo_costo,
    nuevo_precio_venta=None,
    folio_factura=None,
    proveedor=None,
    linea_id=None,
    id_operacion=None,
):
    """
    Actualiza atómicamente el stock actual usando firestore.Increment, actualiza el precio de costo
    y registra el movimiento individual ('proveedor + folio + línea') en la misma transacción.

    Args:
        sku (str): SKU del producto a actualizar.
        cantidad (int): Cantidad de unidades individuales a sumar al stock actual.
        nuevo_costo (float): Nuevo precio de costo unitario del producto.
        nuevo_precio_venta (float | None, optional): Nuevo precio de venta al público (opcional).
        folio_factura (str | None, optional): Folio de la factura asociada.
        proveedor (str | None, optional): Proveedor de la factura asociada.
        linea_id (int | str | None, optional): Identificador o número de línea dentro de la factura.
        id_operacion (str | None, optional): Identidad única de la operación ('PROVEEDOR::FOLIO::L<N>').

    Returns:
        dict: Resultado de la operación con 'exito', 'ya_aplicado', 'id_operacion' y stock resultante.
    """
    try:
        productos_ref = get_productos_collection()
        doc_ref = productos_ref.document(sku)
        cliente_db = getattr(productos_ref, "_client", None)

        folio_limpio = _limpiar_campo(folio_factura)
        proveedor_limpio = _limpiar_campo(proveedor)
        id_op = _resolver_id_operacion(
            id_operacion=id_operacion,
            proveedor=proveedor_limpio,
            folio_factura=folio_limpio,
            linea_id=linea_id,
        )

        movimientos_ref = _obtener_coleccion_movimientos(productos_ref) if id_op else None
        mov_ref = movimientos_ref.document(id_op) if (movimientos_ref is not None and id_op) else None

        registro_movimiento = (
            {
                "id_operacion": id_op,
                "tipo_movimiento": "ACTUALIZAR_STOCK",
                "sku": sku,
                "proveedor": proveedor_limpio,
                "folio": folio_limpio,
                "linea_id": int(linea_id) if str(linea_id or "").isdigit() else linea_id,
                "cantidad_aplicada": int(cantidad),
                "precio_costo": float(nuevo_costo),
                "fecha_aplicacion": firestore.SERVER_TIMESTAMP,
            }
            if id_op
            else None
        )

        def _preparar_actualizacion(datos_actuales: dict, ya_en_movimientos: bool = False) -> dict:
            stock_anterior = int(datos_actuales.get("stock_actual", 0))
            operaciones_previas = list(datos_actuales.get("operaciones_aplicadas") or [])

            if id_op and (ya_en_movimientos or id_op in operaciones_previas):
                return {
                    "ya_aplicado": True,
                    "stock_anterior": stock_anterior,
                    "nuevo_stock": stock_anterior,
                    "actualizaciones": None,
                }

            nuevo_stock = stock_anterior + int(cantidad)
            actualizaciones = {
                "stock_actual": firestore.Increment(int(cantidad)),
                "precio_costo": float(nuevo_costo),
                "actualizado_en": firestore.SERVER_TIMESTAMP,
            }
            if nuevo_precio_venta is not None and float(nuevo_precio_venta) > 0:
                actualizaciones["precio_venta"] = float(nuevo_precio_venta)
            if id_op:
                actualizaciones["operaciones_aplicadas"] = firestore.ArrayUnion([id_op])
            if folio_limpio:
                actualizaciones["folios_aplicados"] = firestore.ArrayUnion([folio_limpio])

            return {
                "ya_aplicado": False,
                "stock_anterior": stock_anterior,
                "nuevo_stock": nuevo_stock,
                "actualizaciones": actualizaciones,
            }

        transaccion = (
            cliente_db.transaction()
            if (cliente_db is not None and hasattr(cliente_db, "transaction"))
            else None
        )

        # Camino 1: Transacción atómica real en Firestore (stock + movimiento juntos)
        if isinstance(transaccion, FirestoreTransaction):
            @firestore.transactional
            def _ejecutar_tx(tx):
                # Todas las lecturas deben ocurrir antes que las escrituras en Firestore
                prod_snap = doc_ref.get(transaction=tx)
                if not prod_snap.exists:
                    return None, None
                mov_snap = mov_ref.get(transaction=tx) if mov_ref is not None else None

                datos = prod_snap.to_dict() or {}
                plan = _preparar_actualizacion(datos, ya_en_movimientos=_snapshot_existe(mov_snap))

                if not plan["ya_aplicado"] and plan["actualizaciones"]:
                    tx.update(doc_ref, plan["actualizaciones"])
                    if mov_ref is not None and registro_movimiento is not None:
                        tx.set(mov_ref, registro_movimiento)
                return datos, plan

            max_reintentos = 3
            ultimo_error_tx = None
            datos_actuales = None
            plan = None

            for intento in range(1, max_reintentos + 1):
                try:
                    tx_activa = cliente_db.transaction() if intento > 1 else transaccion
                    datos_actuales, plan = _ejecutar_tx(tx_activa)
                    ultimo_error_tx = None
                    break
                except Exception as e_tx:
                    ultimo_error_tx = e_tx
                    if intento < max_reintentos:
                        time.sleep(0.1 * (2 ** (intento - 1)))

            if ultimo_error_tx is not None:
                # JAMÁS degradar a escritura directa no atómica: se conserva la transacción
                # y se reporta el movimiento como fallido/pendiente para auditoría y reintento
                return {
                    "exito": False,
                    "error_transaccional": True,
                    "id_operacion": id_op,
                    "error": str(ultimo_error_tx),
                    "mensaje": (
                        f"Error transaccional al actualizar stock de '{sku}' tras {max_reintentos} "
                        f"reintentos controlados: {ultimo_error_tx}. Transacción abortada para evitar escrituras no atómicas."
                    ),
                }
        # Camino 2: Entorno de pruebas unitarias con colecciones simuladas (mocks/fakes)
        else:
            doc = doc_ref.get()
            if not _snapshot_existe(doc):
                return {
                    "exito": False,
                    "id_operacion": id_op,
                    "mensaje": f"No se encontró el producto con SKU '{sku}' para actualizar.",
                }
            mov_snap = mov_ref.get() if mov_ref is not None else None
            datos_actuales = doc.to_dict() or {}
            plan = _preparar_actualizacion(datos_actuales, ya_en_movimientos=_snapshot_existe(mov_snap))
            if not plan["ya_aplicado"] and plan["actualizaciones"]:
                doc_ref.update(plan["actualizaciones"])
                if mov_ref is not None and registro_movimiento is not None:
                    mov_ref.set(registro_movimiento)

        if datos_actuales is None or plan is None:
            return {
                "exito": False,
                "id_operacion": id_op,
                "mensaje": f"No se encontró el producto con SKU '{sku}' para actualizar.",
            }

        stock_anterior = plan["stock_anterior"]
        nuevo_stock = plan["nuevo_stock"]

        if plan["ya_aplicado"]:
            mensaje = (
                f"Movimiento '{id_op}' ya había sido aplicado previamente a '{sku}'. "
                f"Se omite repetición (stock se mantiene en {nuevo_stock} unidades)."
            )
        else:
            mensaje = f"Stock de '{sku}' actualizado atómicamente de {stock_anterior} a {nuevo_stock} unidades."

        return {
            "exito": True,
            "ya_aplicado": plan["ya_aplicado"],
            "id_operacion": id_op,
            "sku": sku,
            "nombre": datos_actuales.get("nombre", sku),
            "stock_anterior": stock_anterior,
            "nuevo_stock": nuevo_stock,
            "nuevo_costo": nuevo_costo,
            "precio_venta": nuevo_precio_venta or datos_actuales.get("precio_venta"),
            "mensaje": mensaje,
        }
    except Exception as e:
        return {
            "exito": False,
            "id_operacion": _limpiar_campo(id_operacion),
            "error": str(e),
            "mensaje": f"Error al actualizar stock de '{sku}': {e}",
        }


@tool
def registrar_nuevo_producto(
    sku,
    nombre,
    marca,
    categoria,
    subcategoria,
    formato,
    precio_costo,
    precio_venta,
    stock_inicial,
    stock_minimo=6,
    folio_factura=None,
    proveedor=None,
    linea_id=None,
    id_operacion=None,
):
    """
    Da de alta un nuevo producto en Firestore generando su vector de embedding y registrando
    el movimiento ('proveedor + folio + línea') en la misma transacción atómica.

    Args:
        sku (str): SKU único del producto.
        nombre (str): Nombre del producto.
        marca (str): Marca del producto.
        categoria (str): Categoría del producto.
        subcategoria (str): Subcategoría del producto.
        formato (str): Formato del producto (ej. "Botella 750ml").
        precio_costo (float): Precio de costo unitario del producto.
        precio_venta (float): Precio de venta al público unitario del producto.
        stock_inicial (int): Stock inicial en unidades individuales.
        stock_minimo (int, optional): Stock mínimo para alertas. Default es 6.
        folio_factura (str | None, optional): Folio de la factura de origen.
        proveedor (str | None, optional): Proveedor de la factura de origen.
        linea_id (int | str | None, optional): Identificador de línea dentro de la factura.
        id_operacion (str | None, optional): Identidad única de la operación ('PROVEEDOR::FOLIO::L<N>').

    Returns:
        dict: Resultado de la operación con 'exito', 'ya_aplicado', 'id_operacion' y mensaje.
    """
    try:
        productos_ref = get_productos_collection()
        doc_ref = productos_ref.document(sku)
        cliente_db = getattr(productos_ref, "_client", None)

        folio_limpio = _limpiar_campo(folio_factura)
        proveedor_limpio = _limpiar_campo(proveedor)
        id_op = _resolver_id_operacion(
            id_operacion=id_operacion,
            proveedor=proveedor_limpio,
            folio_factura=folio_limpio,
            linea_id=linea_id,
        )

        movimientos_ref = _obtener_coleccion_movimientos(productos_ref) if id_op else None
        mov_ref = movimientos_ref.document(id_op) if (movimientos_ref is not None and id_op) else None

        texto_descriptivo = f"{nombre} {marca} {categoria} {subcategoria} {formato}".strip()
        cache_doc: dict = {}

        def _construir_documento_producto() -> dict:
            if "nuevo_producto" not in cache_doc:
                vector = generar_embedding(texto_descriptivo)
                doc_nuevo = {
                    "sku": sku,
                    "nombre": nombre,
                    "marca": marca,
                    "categoria": categoria,
                    "subcategoria": subcategoria,
                    "formato": formato,
                    "precio_costo": float(precio_costo),
                    "precio_venta": float(precio_venta),
                    "stock_actual": int(stock_inicial),
                    "stock_minimo": int(stock_minimo),
                    "activo": True,
                    "embedding": Vector(vector),
                    "texto_descriptivo": texto_descriptivo,
                    "creado_en": firestore.SERVER_TIMESTAMP,
                    "actualizado_en": firestore.SERVER_TIMESTAMP,
                }
                if id_op:
                    doc_nuevo["operaciones_aplicadas"] = [id_op]
                if folio_limpio:
                    doc_nuevo["folios_aplicados"] = [folio_limpio]
                cache_doc["nuevo_producto"] = doc_nuevo
            return cache_doc["nuevo_producto"]

        registro_movimiento = (
            {
                "id_operacion": id_op,
                "tipo_movimiento": "CREAR_PRODUCTO",
                "sku": sku,
                "proveedor": proveedor_limpio,
                "folio": folio_limpio,
                "linea_id": int(linea_id) if str(linea_id or "").isdigit() else linea_id,
                "cantidad_aplicada": int(stock_inicial),
                "precio_costo": float(precio_costo),
                "fecha_aplicacion": firestore.SERVER_TIMESTAMP,
            }
            if id_op
            else None
        )

        transaccion = (
            cliente_db.transaction()
            if (cliente_db is not None and hasattr(cliente_db, "transaction"))
            else None
        )

        if isinstance(transaccion, FirestoreTransaction):
            @firestore.transactional
            def _ejecutar_tx_crear(tx):
                prod_snap = doc_ref.get(transaction=tx)
                mov_snap = mov_ref.get(transaction=tx) if mov_ref is not None else None

                ops_previas = (
                    list((prod_snap.to_dict() or {}).get("operaciones_aplicadas") or [])
                    if prod_snap.exists
                    else []
                )
                if id_op and (_snapshot_existe(mov_snap) or id_op in ops_previas):
                    return "YA_APLICADO"
                if prod_snap.exists:
                    return "SKU_EXISTENTE"

                tx.set(doc_ref, _construir_documento_producto())
                if mov_ref is not None and registro_movimiento is not None:
                    tx.set(mov_ref, registro_movimiento)
                return "CREADO"

            max_reintentos = 3
            ultimo_error_tx = None
            estado_creacion = None

            for intento in range(1, max_reintentos + 1):
                try:
                    tx_activa = cliente_db.transaction() if intento > 1 else transaccion
                    estado_creacion = _ejecutar_tx_crear(tx_activa)
                    ultimo_error_tx = None
                    break
                except Exception as e_tx:
                    ultimo_error_tx = e_tx
                    if intento < max_reintentos:
                        time.sleep(0.1 * (2 ** (intento - 1)))

            if ultimo_error_tx is not None:
                return {
                    "exito": False,
                    "error_transaccional": True,
                    "id_operacion": id_op,
                    "error": str(ultimo_error_tx),
                    "mensaje": (
                        f"Error transaccional al crear producto '{sku}' tras {max_reintentos} "
                        f"reintentos controlados: {ultimo_error_tx}. Transacción abortada para evitar escrituras no atómicas."
                    ),
                }
        else:
            prod_snap = doc_ref.get()
            mov_snap = mov_ref.get() if mov_ref is not None else None
            existe_prod = _snapshot_existe(prod_snap)
            ops_previas = (
                list((prod_snap.to_dict() or {}).get("operaciones_aplicadas") or [])
                if existe_prod
                else []
            )
            if id_op and (_snapshot_existe(mov_snap) or id_op in ops_previas):
                estado_creacion = "YA_APLICADO"
            elif existe_prod:
                estado_creacion = "SKU_EXISTENTE"
            else:
                doc_ref.set(_construir_documento_producto())
                if mov_ref is not None and registro_movimiento is not None:
                    mov_ref.set(registro_movimiento)
                estado_creacion = "CREADO"

        if estado_creacion == "YA_APLICADO":
            return {
                "exito": True,
                "ya_aplicado": True,
                "id_operacion": id_op,
                "sku": sku,
                "nombre": nombre,
                "stock_inicial": stock_inicial,
                "mensaje": f"Movimiento '{id_op}' ya había creado previamente '{sku}'. Se omite duplicación.",
            }

        if estado_creacion == "SKU_EXISTENTE":
            return {
                "exito": False,
                "ya_aplicado": False,
                "id_operacion": id_op,
                "mensaje": f"El producto con SKU '{sku}' ya existe en el catálogo.",
            }

        return {
            "exito": True,
            "ya_aplicado": False,
            "id_operacion": id_op,
            "sku": sku,
            "nombre": nombre,
            "stock_inicial": stock_inicial,
            "mensaje": f"Producto '{nombre}' ({sku}) registrado exitosamente con vector semántico.",
        }
    except Exception as e:
        return {
            "exito": False,
            "id_operacion": _limpiar_campo(id_operacion),
            "error": str(e),
            "mensaje": f"Error al registrar producto '{sku}': {e}",
        }


@tool
def registrar_auditoria_factura(
    folio,
    proveedor,
    total_items,
    items_procesados,
    estado="COMPLETADO",
    auditoria_id_existente=None,
):
    """
    Registra o actualiza el procesamiento de una factura en la colección 'historial_facturas' de Firestore
    como memoria episódica/transaccional a largo plazo para auditoría y control de duplicados.

    Args:
        folio (str): Folio o número de la factura.
        proveedor (str): Nombre del proveedor de la factura.
        total_items (int): Total de ítems consolidados en la factura.
        items_procesados (list[dict]): Lista de ítems procesados con sus detalles y estados por línea.
        estado (str, optional): Estado del procesamiento ('COMPLETADO' o 'COMPLETADO_CON_ERRORES').
        auditoria_id_existente (str | None, optional): ID de auditoría previa cuando se completa un reintento parcial.

    Returns:
        dict: Diccionario con el resultado de la operación, incluyendo el ID de auditoría y mensaje.
    """
    historial_ref = get_historial_collection()

    items_ordenados = sorted(
        list(items_procesados or []),
        key=lambda x: x.get("linea_id", 10**6) if isinstance(x, dict) else 10**6,
    )

    registro = {
        "folio": str(folio),
        "proveedor": str(proveedor),
        "total_items": total_items,
        "items": items_ordenados,
        "estado": estado,
        "fecha_procesamiento": firestore.SERVER_TIMESTAMP,
    }

    aud_previa = _limpiar_campo(auditoria_id_existente)
    if aud_previa:
        historial_ref.document(aud_previa).set(registro)
        id_registro = aud_previa
    else:
        doc_ref = historial_ref.add(registro)
        id_registro = doc_ref[1].id

    return {
        "exito": True,
        "id_auditoria": id_registro,
        "mensaje": f"Auditoría de factura folio '{folio}' guardada con ID: {id_registro} ({estado}).",
    }


def normalizar_texto_equivalencia(valor: Any) -> str:
    """Normaliza proveedor o glosa para comparación consistente (mayúsculas y espacios simples)."""
    limpio = _limpiar_campo(valor)
    if not limpio or limpio.lower() == "ilegible":
        return ""
    return re.sub(r"\s+", " ", limpio.upper()).strip()


def construir_clave_equivalencia(proveedor: Any, glosa: Any) -> Optional[str]:
    """
    Construye el ID determinístico en Firestore para recordar la equivalencia aprobada
    entre un proveedor y una glosa específica ('<PROVEEDOR>::<GLOSA_SLUG>_<HASH8>').
    """
    glosa_norm = normalizar_texto_equivalencia(glosa)
    if not glosa_norm:
        return None
    prov_norm = normalizar_texto_equivalencia(proveedor) or "SIN_PROVEEDOR"
    digest = hashlib.sha1(f"{prov_norm}||{glosa_norm}".encode("utf-8")).hexdigest()[:8]
    prov_seg = _normalizar_segmento_id(prov_norm, "SIN_PROVEEDOR")[:40]
    glosa_seg = _normalizar_segmento_id(glosa_norm, "GLOSA")[:60]
    return f"{prov_seg}::{glosa_seg}_{digest}"


@tool
def guardar_equivalencia_proveedor(
    proveedor,
    glosa,
    sku,
    formato=None,
    origen="APROBACION_HUMANA",
):
    """
    Guarda en la memoria de largo plazo ('equivalencias_proveedor' en Firestore) la equivalencia
    aprobada entre un proveedor, una glosa de factura y el SKU correspondiente del catálogo,
    para recuperarla automáticamente en futuras facturas del mismo proveedor.

    Args:
        proveedor (str | None): Nombre del proveedor de la factura.
        glosa (str): Glosa comercial tal como aparece en la factura.
        sku (str): SKU del catálogo aprobado para esa glosa.
        formato (str | None, optional): Formato detectado de la línea (opcional).
        origen (str, optional): Fuente de la aprobación (default 'APROBACION_HUMANA').

    Returns:
        dict: Resultado con 'exito', 'clave_equivalencia' y mensaje.
    """
    sku_limpio = _limpiar_campo(sku)
    clave = construir_clave_equivalencia(proveedor, glosa)
    if not clave or not sku_limpio:
        return {
            "exito": False,
            "clave_equivalencia": None,
            "mensaje": "No se puede guardar equivalencia sin una glosa legible y un SKU válido.",
        }

    try:
        equivalencias_ref = get_equivalencias_collection()
        prov_norm = normalizar_texto_equivalencia(proveedor) or "SIN_PROVEEDOR"
        glosa_norm = normalizar_texto_equivalencia(glosa)

        registro = {
            "clave_equivalencia": clave,
            "proveedor": _limpiar_campo(proveedor) or "SIN_PROVEEDOR",
            "proveedor_normalizado": prov_norm,
            "glosa_original": str(glosa).strip(),
            "glosa_normalizada": glosa_norm,
            "formato": _limpiar_campo(formato),
            "sku_homologado": sku_limpio,
            "origen": origen,
            "actualizado_en": firestore.SERVER_TIMESTAMP,
        }
        equivalencias_ref.document(clave).set(registro)
        return {
            "exito": True,
            "clave_equivalencia": clave,
            "sku_homologado": sku_limpio,
            "mensaje": f"Equivalencia aprendida guardada: ('{prov_norm}', '{glosa_norm}') -> '{sku_limpio}'.",
        }
    except Exception as e:
        return {
            "exito": False,
            "clave_equivalencia": clave,
            "error": str(e),
            "mensaje": f"No se pudo guardar la equivalencia en Firestore: {e}",
        }

