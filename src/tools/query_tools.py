"""
Herramientas de Consulta (Query Tools) para el agente.
Permiten recuperar información de Firestore: duplicados (distinguiendo facturas completadas de
facturas con errores parciales), búsqueda vectorial semántica e inventario actual.
"""
from langchain_core.tools import tool
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.base_query import FieldFilter

from src.services.firebase import (
    get_productos_collection,
    get_historial_collection,
    get_equivalencias_collection,
)
from src.services.embeddings import generar_embedding
from src.services.gemini import (
    extraer_datos_factura,
    validar_esquema_extraccion,
    FacturaExtraidaSchema,
    ItemFacturaExtraido,
)
from src.tools.write_tools import construir_clave_equivalencia


_ESTADOS_REINTENTABLES = {
    "COMPLETADO_CON_ERRORES",
    "PARCIAL",
    "ERROR",
    "ERROR_EXTRACCION",
    "PENDIENTE_REVISION",
}


@tool
def extraer_factura_desde_imagen(ruta_imagen):
    """
    Extrae de forma estructurada los datos de una factura de compra (proveedor, número de folio
    y lista de ítems detallados con glosa, formato, cantidad y precio unitario) a partir de una
    imagen (PNG o JPG) utilizando visión multimodal con Google Gemini, validando contra esquema Pydantic.

    Args:
        ruta_imagen (str): Ruta local o URL de la imagen de la factura a procesar.

    Returns:
        dict: Diccionario validado con 'proveedor', 'numero_factura', 'items' y 'estado_extraccion'.
    """
    try:
        resultado = extraer_datos_factura(ruta_imagen)
        return validar_esquema_extraccion(resultado)
    except Exception as e:
        return validar_esquema_extraccion({
            "proveedor": None,
            "numero_factura": None,
            "items": [],
            "tipo_documento": "ILEGIBLE",
            "es_documento_valido": False,
            "motivo_sin_items": f"Fallo al procesar imagen: {e}",
            "estado_extraccion": "ERROR_EXTRACCION",
        })


@tool
def consultar_duplicidad_factura(folio=None, proveedor=None):
    """
    Verifica en el historial de Firestore si la factura ya fue procesada anteriormente.
    Distingue entre facturas finalizadas con éxito ('COMPLETADO', que se bloquean como duplicadas)
    y facturas previas con fallos parciales ('COMPLETADO_CON_ERRORES', que permiten retomar
    únicamente sus líneas fallidas sin repetir las exitosas).

    Args:
        folio (str | None): Folio o número de la factura a verificar (opcional).
        proveedor (str | None): Nombre del proveedor de la factura a verificar (opcional).

    Returns:
        dict: Diccionario con 'es_duplicada', 'estado_verificacion' y 'registro_existente' si aplica.
    """
    # Caso 1: Folio no especificado o ilegible -> se continúa sin verificar duplicidad
    folio_limpio = str(folio).strip() if folio is not None else ""
    if not folio_limpio or folio_limpio.lower() in ["none", "null", "s/f", "n/a"]:
        return {
            "es_duplicada": False,
            "estado_verificacion": "FOLIO_ILEGIBLE",
            "mensaje": "Folio no especificado o ilegible. Se continúa sin verificar duplicidad.",
        }

    try:
        historial_ref = get_historial_collection()
        query = historial_ref.where(filter=FieldFilter("folio", "==", folio_limpio))

        proveedor_limpio = str(proveedor).strip() if proveedor is not None else ""
        if proveedor_limpio and proveedor_limpio.lower() not in ["none", "null", "desconocido", "n/a"]:
            query = query.where(filter=FieldFilter("proveedor", "==", proveedor_limpio))

        docs = list(query.stream())
        if docs:
            doc_completado = None
            datos_completado = None
            doc_parcial = None
            datos_parcial = None

            for doc in docs:
                datos_doc = doc.to_dict() or {}
                estado_doc = str(datos_doc.get("estado") or "COMPLETADO").strip().upper()
                if estado_doc not in _ESTADOS_REINTENTABLES:
                    doc_completado = doc
                    datos_completado = datos_doc
                    break
                doc_parcial = doc
                datos_parcial = datos_doc

            # Si ya existe un registro COMPLETADO sin errores -> bloquear como duplicada
            if doc_completado is not None and datos_completado is not None:
                return {
                    "es_duplicada": True,
                    "estado_verificacion": "DUPLICADA",
                    "mensaje": (
                        f"La factura folio '{folio_limpio}' de "
                        f"'{proveedor_limpio or 'Proveedor desconocido'}' ya fue registrada y completada previamente."
                    ),
                    "registro_existente": {
                        "id": doc_completado.id,
                        "estado": datos_completado.get("estado", "COMPLETADO"),
                        "fecha": str(datos_completado.get("fecha_procesamiento")),
                        "total_items": datos_completado.get("total_items", 0),
                    },
                }

            # Si solo existe registro previo con errores (COMPLETADO_CON_ERRORES) -> permitir reintento selectivo
            if doc_parcial is not None and datos_parcial is not None:
                estado_previo = datos_parcial.get("estado", "COMPLETADO_CON_ERRORES")
                return {
                    "es_duplicada": False,
                    "es_reintento_parcial": True,
                    "estado_verificacion": "REINTENTO_CON_ERRORES",
                    "mensaje": (
                        f"La factura folio '{folio_limpio}' tiene un registro previo en estado "
                        f"'{estado_previo}' (ID: {doc_parcial.id}). Se retomarán solo las líneas fallidas."
                    ),
                    "registro_existente": {
                        "id": doc_parcial.id,
                        "estado": estado_previo,
                        "fecha": str(datos_parcial.get("fecha_procesamiento")),
                        "total_items": datos_parcial.get("total_items", 0),
                        "items": list(datos_parcial.get("items") or []),
                    },
                }

        # Caso 2: Folio legible y verificado sin coincidencias
        return {
            "es_duplicada": False,
            "estado_verificacion": "VERIFICADA_NUEVA",
            "mensaje": f"Factura folio '{folio_limpio}' es nueva y no está duplicada.",
        }
    except Exception as e:
        # Caso 3: Error real de conexión o consulta a Firestore -> reportar fallo de verificación
        return {
            "es_duplicada": None,
            "estado_verificacion": "ERROR_CONSULTA",
            "error": str(e),
            "mensaje": f"Error al consultar duplicidad en Firestore: {e}",
        }


@tool
def buscar_candidatos_vectoriales(glosa=None, formato=None, top_k=3):
    """
    Realiza una búsqueda semántica de los productos más cercanos en Firestore
    comparando el vector de la glosa de la factura contra los embeddings del catálogo interno.

    Args:
        glosa (str | None): Glosa de la línea de factura a buscar.
        formato (str | None): Formato detectado de la línea de factura (opcional).
        top_k (int, optional): Número máximo de candidatos a retornar. Default es 3.

    Returns:
        list[dict]: Lista de diccionarios con los productos candidatos más similares.
    """
    glosa_limpia = str(glosa).strip() if glosa and str(glosa).strip().lower() != "null" else ""
    formato_limpio = str(formato).strip() if formato and str(formato).strip().lower() != "null" else ""

    if not glosa_limpia:
        return []

    texto_busqueda = f"{glosa_limpia} {formato_limpio}".strip()
    vector_query = generar_embedding(texto_busqueda)

    productos_ref = get_productos_collection()
    try:
        query_vectorial = productos_ref.find_nearest(
            vector_field="embedding",
            query_vector=Vector(vector_query),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
            distance_result_field="distancia_coseno",
        )
    except TypeError:
        query_vectorial = productos_ref.find_nearest(
            vector_field="embedding",
            query_vector=Vector(vector_query),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
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
            "stock_actual": data.get("stock_actual", 0),
        }
        distancia = data.get("distancia_coseno")
        if isinstance(distancia, (int, float)):
            item_cand["similitud_vectorial"] = round(max(0.0, min(1.0, 1.0 - float(distancia))), 4)
        candidatos.append(item_cand)

    return candidatos


@tool
def consultar_producto_por_sku(sku):
    """
    Obtiene los datos completos de un producto específico en Firestore mediante su SKU.

    Args:
        sku (str): SKU del producto a consultar.

    Returns:
        dict: Diccionario con la clave 'encontrado' (True/False) y los datos del producto si existe.
    """
    productos_ref = get_productos_collection()
    doc = productos_ref.document(sku).get()
    if doc.exists:
        data = doc.to_dict()
        data["sku"] = doc.id
        data.pop("embedding", None)
        return {"encontrado": True, "producto": data}
    return {"encontrado": False, "mensaje": f"No se encontró el producto con SKU '{sku}'"}


@tool
def consultar_equivalencia_proveedor(proveedor=None, glosa=None):
    """
    Consulta en la memoria de largo plazo ('equivalencias_proveedor' en Firestore) si ya existe
    una equivalencia previamente aprobada para el par (proveedor, glosa).

    Args:
        proveedor (str | None): Nombre del proveedor de la factura.
        glosa (str | None): Glosa comercial de la línea de factura.

    Returns:
        dict: Diccionario con 'encontrada' (True/False), 'sku_homologado' y detalles de la equivalencia.
    """
    clave = construir_clave_equivalencia(proveedor, glosa)
    if not clave:
        return {"encontrada": False, "sku_homologado": None}

    try:
        equivalencias_ref = get_equivalencias_collection()
        doc = equivalencias_ref.document(clave).get()
        if getattr(doc, "exists", False) is True:
            datos = doc.to_dict() or {}
            sku = datos.get("sku_homologado")
            if sku:
                return {
                    "encontrada": True,
                    "clave_equivalencia": clave,
                    "sku_homologado": sku,
                    "equivalencia": datos,
                }
        return {"encontrada": False, "clave_equivalencia": clave, "sku_homologado": None}
    except Exception as e:
        return {
            "encontrada": False,
            "clave_equivalencia": clave,
            "sku_homologado": None,
            "error": str(e),
        }

