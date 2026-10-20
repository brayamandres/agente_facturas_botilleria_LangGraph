"""
Herramientas de Escritura (Writing Tools) para el agente.
Permiten persistir cambios en Firestore: actualizar stock, crear nuevos productos e insertar auditoría.
"""
from langchain_core.tools import tool
from firebase_admin import firestore
from google.cloud.firestore_v1.vector import Vector

from src.services.firebase import get_productos_collection, get_historial_collection
from src.services.embeddings import generar_embedding

#TOOLS

#PARA ACTUALIZAR FIREBASE
@tool
def actualizar_stock_y_costo(
    sku: str,
    cantidad: int,
    nuevo_costo: float,
    nuevo_precio_venta: float = None
) -> dict:
    """
    Actualiza el stock actual sumando la cantidad comprada y actualiza el precio de costo del producto en Firestore.
    Opcionalmente puede actualizar también el precio de venta al público.
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
    sku: str,
    nombre: str,
    marca: str,
    categoria: str,
    subcategoria: str,
    formato: str,
    precio_costo: float,
    precio_venta: float,
    stock_inicial: int,
    stock_minimo: int = 6
) -> dict:
    """
    Da de alta un nuevo producto en Firestore generando su vector de embedding para que
    pueda ser localizado en futuras búsquedas semánticas.
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
    folio: str,
    proveedor: str,
    total_items: int,
    items_procesados: list[dict],
    estado: str = "COMPLETADO"
) -> dict:
    """
    Registra el procesamiento de una factura en la colección 'historial_facturas' de Firestore
    como memoria episódica/transaccional a largo plazo para auditoría y control de duplicados.
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

