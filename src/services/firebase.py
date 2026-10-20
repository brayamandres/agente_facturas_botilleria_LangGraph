import firebase_admin
from firebase_admin import credentials, firestore
from src.config import FIREBASE_CREDENTIALS_PATH, BASE_DIR
import os

_firestore_client = None

def get_firestore_client():
    """
    Retorna la instancia del cliente de Firestore (Singleton).
    Inicializa Firebase Admin SDK si no ha sido inicializado.
    """
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    if not FIREBASE_CREDENTIALS_PATH:
        raise ValueError(
            "FIREBASE_CREDENTIALS_PATH no está configurado en .env. "
            "Por favor especifica la ruta al archivo .json de credenciales de Firebase."
        )

    # Permitir ruta relativa o absoluta
    cred_file = FIREBASE_CREDENTIALS_PATH
    if not os.path.isabs(cred_file):
        cred_file = str(BASE_DIR / cred_file)

    if not os.path.exists(cred_file):
        raise FileNotFoundError(
            f"No se encontró el archivo de credenciales de Firebase en: {cred_file}"
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_file)
        firebase_admin.initialize_app(cred)

    _firestore_client = firestore.client()
    return _firestore_client


def get_productos_collection():
    """Retorna la referencia a la colección 'productos'."""
    db = get_firestore_client()
    return db.collection("productos")


def get_historial_collection():
    """Retorna la referencia a la colección 'historial_facturas' para memoria de largo plazo."""
    db = get_firestore_client()
    return db.collection("historial_facturas")

