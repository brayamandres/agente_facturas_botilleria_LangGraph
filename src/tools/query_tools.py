"""
Herramientas de Consulta (Query Tools) para el agente.
Permiten recuperar información de Firestore: duplicados, búsqueda vectorial semántica e inventario actual.
"""
from langchain_core.tools import tool
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.base_query import FieldFilter

from src.services.firebase import get_productos_collection, get_historial_collection
from src.services.embeddings import generar_embedding
from src.services.gemini import extraer_datos_factura


@tool
def extraer_factura_desde_imagen(ruta_imagen):
    """
    Extrae de forma estructurada los datos de una factura de compra (proveedor, número de folio
    y lista de ítems detallados con glosa, formato, cantidad y precio unitario) a partir de una
    imagen (PNG o JPG) utilizando visión multimodal con Google Gemini.
    
    Args:
        ruta_imagen (str): Ruta local o URL de la imagen de la factura a procesar.
        
    Returns:
        dict: Diccionario con los datos extraídos de la factura, incluyendo proveedor, folio y lista de ítems.
    """
    # Llama al servicio de Gemini para procesar visualmente la factura y devolver un JSON estructurado
    return extraer_datos_factura(ruta_imagen)


@tool
def consultar_duplicidad_factura(folio = None, proveedor = None):
    """
    Verifica en el historial de Firestore si la factura ya fue procesada anteriormente
    para evitar doble contabilización en el inventario.

    Args:
        folio (str | None): Folio o número de la factura a verificar (opcional).
        proveedor (str | None): Nombre del proveedor de la factura a verificar (opcional).
        
    Returns:
        dict: Diccionario con "es_duplicada", "estado_verificacion" y un mensaje explicativo.
    """
    # Caso 1: Folio no especificado o ilegible -> se continúa sin verificar duplicidad
    folio_limpio = str(folio).strip() if folio is not None else ""
    if not folio_limpio or folio_limpio.lower() in ["none", "null", "s/f", "n/a"]:
        return {
            "es_duplicada": False,
            "estado_verificacion": "FOLIO_ILEGIBLE",
            "mensaje": "Folio no especificado o ilegible. Se continúa sin verificar duplicidad."
        }

    try:
        # Se guarda en una variable la colección histórica de facturas ingresadas
        historial_ref = get_historial_collection()
        query = historial_ref.where(filter=FieldFilter("folio", "==", folio_limpio))

        proveedor_limpio = str(proveedor).strip() if proveedor is not None else ""
        if proveedor_limpio and proveedor_limpio.lower() not in ["none", "null", "desconocido", "n/a"]:
            query = query.where(filter=FieldFilter("proveedor", "==", proveedor_limpio))

        # Se envía la consulta y los resultados se almacenan
        docs = list(query.stream())
        if docs:
            datos = docs[0].to_dict()
            return {
                "es_duplicada": True,
                "estado_verificacion": "DUPLICADA",
                "mensaje": f"La factura folio '{folio_limpio}' de '{proveedor_limpio or 'Proveedor desconocido'}' ya fue registrada previamente.",
                "registro_existente": {
                    "id": docs[0].id,
                    "fecha": str(datos.get("fecha_procesamiento")),
                    "total_items": datos.get("total_items", 0)
                }
            }

        # Caso 2: Folio legible y verificado sin coincidencias
        return {
            "es_duplicada": False,
            "estado_verificacion": "VERIFICADA_NUEVA",
            "mensaje": f"Factura folio '{folio_limpio}' es nueva y no está duplicada."
        }
    except Exception as e:
        # Caso 3: Error real de conexión o consulta a Firestore -> reportar fallo de verificación
        return {
            "es_duplicada": None,
            "estado_verificacion": "ERROR_CONSULTA",
            "error": str(e),
            "mensaje": f"Error al consultar duplicidad en Firestore: {e}"
        }


@tool
def buscar_candidatos_vectoriales(glosa = None, formato = None, top_k = 3):
    """
    Realiza una búsqueda semántica de los productos más cercanos en Firestore
    comparando el vector de la glosa de la factura contra los embeddings del catálogo interno.
    
    Args:
        glosa (str | None): Glosa de la línea de factura a buscar.
        formato (str | None): Formato detectado de la línea de factura (opcional).
        top_k (int, optional): Número máximo de candidatos a retornar. Default es 3.
    
    Returns:
        list[dict]: Lista de diccionarios con los productos candidatos más similares, incluyendo SKU, nombre, marca, formato y precios actuales.
    """
    # Se formatean las entradas y se vectorizan de forma segura ante valores nulos/ilegibles
    glosa_limpia = str(glosa).strip() if glosa and str(glosa).strip().lower() != "null" else ""
    formato_limpio = str(formato).strip() if formato and str(formato).strip().lower() != "null" else ""
    
    if not glosa_limpia:
        return []

    texto_busqueda = f"{glosa_limpia} {formato_limpio}".strip()
    vector_query = generar_embedding(texto_busqueda)

    # Se extraen los k=3 productos vectorialmente más similares
    productos_ref = get_productos_collection()
    try:
        query_vectorial = productos_ref.find_nearest(
            vector_field="embedding",
            query_vector=Vector(vector_query),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
            distance_result_field="distancia_coseno"
        )
    except TypeError:
        query_vectorial = productos_ref.find_nearest(
            vector_field="embedding",
            query_vector=Vector(vector_query),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k
        )

    candidatos = []
    for doc in query_vectorial.get():
        data = doc.to_dict()
        item_cand = {
            "sku": doc.id,
            "nombre": data.get("nombre", ""),
            "marca": data.get("marca", ""),
            "formato": data.get("formato", ""),
            "precio_costo_actual": data.get("precio_costo", 0),
            "precio_venta_actual": data.get("precio_venta", 0),
            "stock_actual": data.get("stock_actual", 0)
        }
        distancia = data.get("distancia_coseno")
        if isinstance(distancia, (int, float)):
            item_cand["similitud_vectorial"] = round(max(0.0, min(1.0, 1.0 - float(distancia))), 4)
        candidatos.append(item_cand)

    # Retorna los 3 productos candidatos con sus características organizado en JSON
    return candidatos


@tool
def consultar_producto_por_sku(sku):
    """
    Obtiene los datos completos de un producto específico en Firestore mediante su SKU.
    
    Args:
        sku (str): SKU del producto a consultar.
    
    Returns:
        dict: Diccionario con la clave "encontrado" (True/False) y los datos del producto si se encuentra, o un mensaje de error si no.
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
