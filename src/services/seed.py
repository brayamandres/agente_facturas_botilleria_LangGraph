"""
Script de Inicialización y Siembra (Seed) del Catálogo en Firestore con Embeddings.
Carga los 40 productos iniciales de la botillería con sus vectores de 384 dimensiones.
"""
import json
from pathlib import Path
from firebase_admin import firestore
from google.cloud.firestore_v1.vector import Vector

from src.services.firebase import get_productos_collection
from src.services.embeddings import generar_embedding

DATA_FILE = Path(__file__).resolve().parent / "seed_data.json"


def sembrar_catalogo_inicial(forzar_recalculo: bool = False):
    """
    Inserta o actualiza los 40 productos iniciales en Firestore.
    Calcula los embeddings vectoriales para habilitar la búsqueda semántica.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        productos = json.load(f)

    coleccion = get_productos_collection()
    print(f"\n[Seed] Iniciando siembra de {len(productos)} productos en Firestore...")

    contador_insertados = 0
    contador_omitidos = 0

    for i, prod in enumerate(productos, 1):
        sku = prod["sku"]
        doc_ref = coleccion.document(sku)
        doc = doc_ref.get()

        if doc.exists and not forzar_recalculo:
            datos_existentes = doc.to_dict()
            if "embedding" in datos_existentes:
                contador_omitidos += 1
                continue

        # Generar texto descriptivo y vector de embedding
        texto = f"{prod.get('nombre', '')} {prod.get('marca', '')} {prod.get('categoria', '')} {prod.get('subcategoria', '')} {prod.get('formato', '')}".strip()
        vector = generar_embedding(texto)

        payload = prod.copy()
        payload["embedding"] = Vector(vector)
        payload["texto_descriptivo"] = texto
        payload["creado_en"] = firestore.SERVER_TIMESTAMP
        payload["actualizado_en"] = firestore.SERVER_TIMESTAMP

        doc_ref.set(payload, merge=True)
        contador_insertados += 1
        print(f" [{i}/{len(productos)}] Producto indexado: {sku} - {prod.get('nombre')}")

    print(f"\n[Seed] Proceso completado exitosamente:")
    print(f" -> {contador_insertados} productos insertados/actualizados con vector.")
    print(f" -> {contador_omitidos} productos ya estaban indexados previamente.\n")


if __name__ == "__main__":
    sembrar_catalogo_inicial()

