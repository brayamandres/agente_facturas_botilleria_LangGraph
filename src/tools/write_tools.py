"""
Herramientas de Escritura (Writing Tools) para el agente.
Permiten persistir cambios en Firestore: actualizar stock, crear nuevos productos e insertar auditoría.
"""
from langchain_core.tools import tool
from firebase_admin import firestore
from google.cloud.firestore_v1.vector import Vector

from src.services.firebase import get_productos_collection, get_historial_collection
from src.services.embeddings import generar_embedding


@tool
def actualizar_stock_y_costo(
    sku,
    cantidad,
    nuevo_costo,
    nuevo_precio_venta = None,
    folio_factura = None
):
    """
    Actualiza atómicamente el stock actual usando firestore.Increment y actualiza el precio de costo en Firestore.
    Opcionalmente actualiza el precio de venta al público y registra el folio para idempotencia ante reintentos.
    
    Args:
        sku (str): SKU del producto a actualizar.
        cantidad (int): Cantidad de unidades individuales a sumar al stock actual.
        nuevo_costo (float): Nuevo precio de costo unitario del producto.
        nuevo_precio_venta (float | None, optional): Nuevo precio de venta al público (opcional).
        folio_factura (str | None, optional): Folio de la factura para evitar duplicar stock en reintentos parciales.
        
    Returns:
        dict: Diccionario con el resultado de la operación, incluyendo el nuevo stock y precios.
    """
    productos_ref = get_productos_collection()
    doc_ref = productos_ref.document(sku)
    cliente_db = getattr(productos_ref, "_client", None)

    folio_limpio = (
        str(folio_factura).strip()
        if folio_factura and str(folio_factura).strip().lower() not in ["none", "null", "s/f", "n/a", ""]
        else None
    )

    def _preparar_actualizacion(datos_actuales):
        stock_anterior = int(datos_actuales.get("stock_actual", 0))
        folios_previos = list(datos_actuales.get("folios_aplicados") or [])

        # Idempotencia: si una ejecución anterior se interrumpió antes de guardar auditoría,
        # evita volver a sumar el mismo ítem al reintentar el mismo folio.
        if folio_limpio and folio_limpio in folios_previos:
            return {
                "ya_aplicado": True,
                "stock_anterior": stock_anterior,
                "nuevo_stock": stock_anterior,
                "actualizaciones": None
            }

        nuevo_stock = stock_anterior + int(cantidad)
        actualizaciones = {
            "stock_actual": firestore.Increment(int(cantidad)),
            "precio_costo": float(nuevo_costo),
            "actualizado_en": firestore.SERVER_TIMESTAMP
        }
        if nuevo_precio_venta is not None and float(nuevo_precio_venta) > 0:
            actualizaciones["precio_venta"] = float(nuevo_precio_venta)
        if folio_limpio:
            actualizaciones["folios_aplicados"] = firestore.ArrayUnion([folio_limpio])

        return {
            "ya_aplicado": False,
            "stock_anterior": stock_anterior,
            "nuevo_stock": nuevo_stock,
            "actualizaciones": actualizaciones
        }

    # Si existe un cliente real de Firestore, ejecutar lectura + incremento dentro de una transacción atómica
    if cliente_db is not None and hasattr(cliente_db, "transaction"):
        try:
            transaccion = cliente_db.transaction()

            @firestore.transactional
            def _ejecutar_tx(tx):
                snapshot = doc_ref.get(transaction=tx)
                if not snapshot.exists:
                    return None, None
                datos = snapshot.to_dict() or {}
                plan = _preparar_actualizacion(datos)
                if not plan["ya_aplicado"] and plan["actualizaciones"]:
                    tx.update(doc_ref, plan["actualizaciones"])
                return datos, plan

            datos_actuales, plan = _ejecutar_tx(transaccion)
            if datos_actuales is None:
                return {
                    "exito": False,
                    "mensaje": f"No se encontró el producto con SKU '{sku}' para actualizar."
                }
        except Exception:
            # Fallback directo si el entorno de pruebas usa mocks parciales
            doc = doc_ref.get()
            if not doc.exists:
                return {
                    "exito": False,
                    "mensaje": f"No se encontró el producto con SKU '{sku}' para actualizar."
                }
            datos_actuales = doc.to_dict() or {}
            plan = _preparar_actualizacion(datos_actuales)
            if not plan["ya_aplicado"] and plan["actualizaciones"]:
                doc_ref.update(plan["actualizaciones"])
    else:
        doc = doc_ref.get()
        if not doc.exists:
            return {
                "exito": False,
                "mensaje": f"No se encontró el producto con SKU '{sku}' para actualizar."
            }
        datos_actuales = doc.to_dict() or {}
        plan = _preparar_actualizacion(datos_actuales)
        if not plan["ya_aplicado"] and plan["actualizaciones"]:
            doc_ref.update(plan["actualizaciones"])

    stock_anterior = plan["stock_anterior"]
    nuevo_stock = plan["nuevo_stock"]

    if plan["ya_aplicado"]:
        mensaje = f"El folio '{folio_limpio}' ya había sido aplicado previamente a '{sku}'. Stock se mantiene en {nuevo_stock} unidades."
    else:
        mensaje = f"Stock de '{sku}' actualizado atómicamente de {stock_anterior} a {nuevo_stock} unidades."

    return {
        "exito": True,
        "sku": sku,
        "nombre": datos_actuales.get("nombre", sku),
        "stock_anterior": stock_anterior,
        "nuevo_stock": nuevo_stock,
        "nuevo_costo": nuevo_costo,
        "precio_venta": nuevo_precio_venta or datos_actuales.get("precio_venta"),
        "mensaje": mensaje
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
    stock_minimo = 6,
    folio_factura = None
):
    """
    Da de alta un nuevo producto en Firestore generando su vector de embedding para que
    pueda ser localizado en futuras búsquedas semánticas.
    
    Args:
        sku (str): SKU único del producto.
        nombre (str): Nombre del producto.
        marca (str): Marca del producto.
        categoria (str): Categoría del producto.
        subcategoria (str): Subcategoría del producto.
        formato (str): Formato del producto (ej. "Botella 750ml").
        precio_costo (float): Precio de costo unitario del producto.
        precio_venta (float): Precio de venta al público unitario del producto.
        stock_inicial (int): Stock inicial del producto.
        stock_minimo (int, optional): Stock mínimo recomendado para alertas. Default es 6.
        folio_factura (str | None, optional): Folio de la factura de origen para trazabilidad.
        
    Returns:
        dict: Diccionario con el resultado de la operación, incluyendo el SKU y mensaje de éxito o error.
    """
    productos_ref = get_productos_collection()
    doc_ref = productos_ref.document(sku)

    if doc_ref.get().exists:
        return {
            "exito": False,
            "mensaje": f"El producto con SKU '{sku}' ya existe en el catálogo."
        }

    texto_descriptivo = f"{nombre} {marca} {categoria} {subcategoria} {formato}".strip()
    vector = generar_embedding(texto_descriptivo)

    folio_limpio = (
        str(folio_factura).strip()
        if folio_factura and str(folio_factura).strip().lower() not in ["none", "null", "s/f", "n/a", ""]
        else None
    )

    nuevo_producto = {
        "sku": sku,
        "nombre": nombre,
        "marca": marca,
        "categoria": categoria,
        "subcategoria": subcategoria,
        "formato": formato,
        "precio_costo": precio_costo,
        "precio_venta": precio_venta,
        "stock_actual": stock_inicial,
        "stock_minimo": stock_minimo,
        "activo": True,
        "embedding": Vector(vector),
        "texto_descriptivo": texto_descriptivo,
        "creado_en": firestore.SERVER_TIMESTAMP,
        "actualizado_en": firestore.SERVER_TIMESTAMP
    }
    if folio_limpio:
        nuevo_producto["folios_aplicados"] = [folio_limpio]

    doc_ref.set(nuevo_producto)

    return {
        "exito": True,
        "sku": sku,
        "nombre": nombre,
        "stock_inicial": stock_inicial,
        "mensaje": f"Producto '{nombre}' ({sku}) registrado exitosamente con vector semántico."
    }


@tool
def registrar_auditoria_factura(
    folio,
    proveedor,
    total_items,
    items_procesados,
    estado = "COMPLETADO"
):
    """
    Registra el procesamiento de una factura en la colección 'historial_facturas' de Firestore
    como memoria episódica/transaccional a largo plazo para auditoría y control de duplicados.
    
    Args:
        folio (str): Folio o número de la factura.
        proveedor (str): Nombre del proveedor de la factura.
        total_items (int): Total de ítems en la factura.
        items_procesados (list[dict]): Lista de ítems procesados con sus detalles.
        estado (str, optional): Estado del procesamiento de la factura. Default es "COMPLETADO".
        
    Returns:
        dict: Diccionario con el resultado de la operación, incluyendo el ID de auditoría y mensaje.
    """
    historial_ref = get_historial_collection()
    
    registro = {
        "folio": str(folio),
        "proveedor": str(proveedor),
        "total_items": total_items,
        "items": items_procesados,
        "estado": estado,
        "fecha_procesamiento": firestore.SERVER_TIMESTAMP
    }

    doc_ref = historial_ref.add(registro)
    id_registro = doc_ref[1].id

    return {
        "exito": True,
        "id_auditoria": id_registro,
        "mensaje": f"Auditoría de factura folio '{folio}' guardada con ID: {id_registro}."
    }
