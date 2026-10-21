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
    nuevo_precio_venta = None):
    """
    Actualiza el stock actual sumando la cantidad comprada y actualiza el precio de costo del producto en Firestore.
    Opcionalmente puede actualizar también el precio de venta al público.
    
    Args:
        sku (str): SKU del producto a actualizar.
        cantidad (int): Cantidad de unidades compradas para sumar al stock actual.
        nuevo_costo (float): Nuevo precio de costo unitario del producto.
        nuevo_precio_venta (float | None, optional): Nuevo precio de venta al público (opcional).
        
    Returns:
        dict: Diccionario con el resultado de la operación, incluyendo el nuevo stock y precios.
    """
    productos_ref = get_productos_collection()
    doc_ref = productos_ref.document(sku)
    doc = doc_ref.get()

    if not doc.exists:
        return {
            "exito": False,
            "mensaje": f"No se encontró el producto con SKU '{sku}' para actualizar."
        }

    datos_actuales = doc.to_dict()
    stock_anterior = datos_actuales.get("stock_actual", 0)
    nuevo_stock = stock_anterior + cantidad

    actualizaciones = {
        "stock_actual": nuevo_stock,
        "precio_costo": nuevo_costo,
        "actualizado_en": firestore.SERVER_TIMESTAMP
    }

    if nuevo_precio_venta is not None and nuevo_precio_venta > 0:
        actualizaciones["precio_venta"] = nuevo_precio_venta

    doc_ref.update(actualizaciones)

    return {
        "exito": True,
        "sku": sku,
        "nombre": datos_actuales.get("nombre", sku),
        "stock_anterior": stock_anterior,
        "nuevo_stock": nuevo_stock,
        "nuevo_costo": nuevo_costo,
        "precio_venta": nuevo_precio_venta or datos_actuales.get("precio_venta"),
        "mensaje": f"Stock de '{sku}' actualizado de {stock_anterior} a {nuevo_stock} unidades."
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
    stock_minimo = 6
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

