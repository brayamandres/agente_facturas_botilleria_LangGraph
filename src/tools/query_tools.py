"""
Herramientas de Consulta (Query Tools) para el agente.
Permiten recuperar información de Firestore: duplicados, búsqueda vectorial semántica e inventario actual.
"""
from langchain_core.tools import tool
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from src.services.firebase import get_productos_collection, get_historial_collection
from src.services.embeddings import generar_embedding
from src.services.gemini import extraer_datos_factura


@tool
def extraer_factura_desde_imagen(ruta_imagen: str) -> dict:
    """
    Extrae de forma estructurada los datos de una factura de compra (proveedor, número de folio
    y lista de ítems detallados con glosa, formato, cantidad y precio unitario) a partir de una
    imagen (PNG o JPG) utilizando visión multimodal con Google Gemini.
    """
    # Llama al servicio de Gemini para procesar visualmente la factura y devolver un JSON estructurado
    return extraer_datos_factura(ruta_imagen)


@tool
def consultar_duplicidad_factura(folio: str, proveedor: str) -> dict:
    """
    Verifica en el historial de Firestore si la factura ya fue procesada anteriormente
    para evitar doble contabilización en el inventario.

    """
    #si es que no se registra el folio de la factura
    if not folio:
        return {"es_duplicada": False, "mensaje": "Folio no especificado, no se puede verificar duplicidad."}

    #se guarda en una variable la colección historica de facturas ingresadas
    historial_ref = get_historial_collection()
    #se comienza a formular la consulta, primero pregunta si el folio nuevo se encuentra en los registros de la bbdd y si se registra proveedor, también se agrega a la consulta
    query = historial_ref.where("folio", "==", str(folio))
    if proveedor:
        query = query.where("proveedor", "==", str(proveedor))

    #se envia la consulta y los resultados se almacenan
    docs = list(query.stream())
    #dependiendo si encuentra o no algo, retorna un Json con la respuesta
    if docs:
        datos = docs[0].to_dict()
        return {
            "es_duplicada": True,
            "mensaje": f"La factura folio '{folio}' de '{proveedor}' ya fue registrada previamente.",
            "registro_existente": {
                "id": docs[0].id,
                "fecha": str(datos.get("fecha_procesamiento")),
                "total_items": datos.get("total_items", 0)
            }
        }
    
    return {"es_duplicada": False, "mensaje": f"Factura folio '{folio}' es nueva y no está duplicada."}


@tool
def buscar_candidatos_vectoriales(glosa: str, formato: str = "", top_k: int = 3) -> list[dict]:
    """
    Realiza una búsqueda semántica de los productos más cercanos en Firestore
    comparando el vector de la glosa de la factura contra los embeddings del catálogo interno.
    """
    #se formatean las entradas y se vectorizan
    texto_busqueda = f"{glosa} {formato or ''}".strip()
    vector_query = generar_embedding(texto_busqueda)

    #se extraen los k=3  productos vectorialmente mas similares
    productos_ref = get_productos_collection()
    query_vectorial = productos_ref.find_nearest(
        vector_field="embedding",
        query_vector=Vector(vector_query),
        distance_measure=DistanceMeasure.COSINE,
        limit=top_k
    )

    candidatos = []
    for doc in query_vectorial.get():
        data = doc.to_dict()
        candidatos.append({
            "sku": doc.id,
            "nombre": data.get("nombre", ""),
            "marca": data.get("marca", ""),
            "formato": data.get("formato", ""),
            "precio_costo_actual": data.get("precio_costo", 0),
            "precio_venta_actual": data.get("precio_venta", 0),
            "stock_actual": data.get("stock_actual", 0)
        })

    #retorna los 3 productos candidatos con sus caracteristicas organizado en json
    return candidatos


@tool
def consultar_producto_por_sku(sku: str) -> dict:
    """
    Obtiene los datos completos de un producto específico en Firestore mediante su SKU.
    """
    productos_ref = get_productos_collection()
    doc = productos_ref.document(sku).get()
    if doc.exists:
        data = doc.to_dict()
        data["sku"] = doc.id
        # Remover el vector para que sea serializable en JSON si se envía al LLM
        data.pop("embedding", None)
        return {"encontrado": True, "producto": data}
    return {"encontrado": False, "mensaje": f"No se encontró el producto con SKU '{sku}'"}

