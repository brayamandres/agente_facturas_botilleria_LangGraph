"""
Servicio de Persistencia de Corto/Mediano Plazo (Checkpointer SQLite) y Gestión de Sesiones.

Permite:
1. Persistir el estado completo del StateGraph (checkpoints, canal de escrituras e interrupciones)
   en un archivo SQLite local ('data/checkpoints.sqlite') para sobrevivir cierres del programa.
2. Registrar y consultar metadatos de sesiones ('thread_id', factura, avance y estado) para
   habilitar la opción 'Reanudar factura pendiente' en la CLI.
"""
import pickle
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from langgraph.checkpoint.memory import MemorySaver

from src.config import CHECKPOINT_DB_PATH

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    _HAS_OFFICIAL_SQLITE_SAVER = True
except ImportError:
    SqliteSaver = None  # type: ignore
    _HAS_OFFICIAL_SQLITE_SAVER = False


class SqlitePersistentSaver(MemorySaver):
    """
    Checkpointer persistente respaldado en SQLite compatible con todas las versiones de LangGraph.
    Se utiliza automáticamente como respaldo si el paquete opcional 'langgraph-checkpoint-sqlite'
    no está instalado en el entorno actual, garantizando persistencia real en disco y seguridad
    multihilo (thread-safety) para interrupt() y Command(resume=...).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__()
        self._conn = conn
        self._lock = threading.RLock()
        with self._lock:
            self._asegurar_esquema()
            self._cargar_desde_sqlite()

    def _asegurar_esquema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS langgraph_checkpoint_store (
                store_key TEXT PRIMARY KEY,
                payload BLOB NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _cargar_desde_sqlite(self) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT payload FROM langgraph_checkpoint_store WHERE store_key = 'global_state'"
            )
            fila = cursor.fetchone()
            if not fila:
                return

            datos = pickle.loads(fila[0])
            storage_guardado = datos.get("storage", {})
            writes_guardados = datos.get("writes", {})
            blobs_guardados = datos.get("blobs", {})

            for thread_id, namespaces in storage_guardado.items():
                for ckpt_ns, ckpts in namespaces.items():
                    self.storage[thread_id][ckpt_ns].update(ckpts)

            for key_w, val_w in writes_guardados.items():
                self.writes[key_w].update(val_w)

            if hasattr(self, "blobs") and isinstance(blobs_guardados, dict):
                self.blobs.update(blobs_guardados)

    def _persistir_en_sqlite(self) -> None:
        with self._lock:
            storage_plano = {
                tid: {ns: dict(cids) for ns, cids in nss.items()}
                for tid, nss in self.storage.items()
            }
            writes_plano = {k: dict(v) for k, v in self.writes.items()}
            blobs_plano = dict(getattr(self, "blobs", {}))

            paquete = pickle.dumps(
                {
                    "storage": storage_plano,
                    "writes": writes_plano,
                    "blobs": blobs_plano,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            ahora = datetime.now().isoformat(timespec="seconds")
            self._conn.execute(
                """
                INSERT INTO langgraph_checkpoint_store (store_key, payload, updated_at)
                VALUES ('global_state', ?, ?)
                ON CONFLICT(store_key) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (paquete, ahora),
            )
            self._conn.commit()

    def get_tuple(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return super().get_tuple(*args, **kwargs)

    def list(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        with self._lock:
            elementos = list(super().list(*args, **kwargs))
        yield from elementos

    def put(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            resultado = super().put(*args, **kwargs)
            self._persistir_en_sqlite()
            return resultado

    def put_writes(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            resultado = super().put_writes(*args, **kwargs)
            self._persistir_en_sqlite()
            return resultado

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            if hasattr(super(), "delete_thread"):
                super().delete_thread(thread_id)  # type: ignore[misc]
            else:
                self.storage.pop(thread_id, None)
                self.writes = type(self.writes)(
                    self.writes.default_factory,
                    {k: v for k, v in self.writes.items() if k[0] != thread_id},
                )
                if hasattr(self, "blobs"):
                    self.blobs = {k: v for k, v in self.blobs.items() if k[0] != thread_id}
            self._persistir_en_sqlite()

    def close(self) -> None:
        """Cierra de forma segura la conexión SQLite subyacente."""
        with self._lock:
            self._conn.close()


_singleton_checkpointer: Any = None
_singleton_db_path: Optional[Path] = None


@contextmanager
def _conectar_sqlite(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Abre y cierra determinísticamente una conexión SQLite asegurando que el directorio exista."""
    ruta = Path(db_path or CHECKPOINT_DB_PATH)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ruta), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_sqlite_checkpointer(db_path: Optional[Path] = None) -> Any:
    """
    Retorna el checkpointer persistente basado en SQLite.
    Si no se especifica un db_path distinto al predeterminado, reutiliza una instancia Singleton.
    """
    global _singleton_checkpointer, _singleton_db_path

    ruta = Path(db_path or CHECKPOINT_DB_PATH).resolve()
    if db_path is None and _singleton_checkpointer is not None and _singleton_db_path == ruta:
        return _singleton_checkpointer

    ruta.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ruta), timeout=30.0, check_same_thread=False)

    if _HAS_OFFICIAL_SQLITE_SAVER and SqliteSaver is not None:
        saver = SqliteSaver(conn)
        if hasattr(saver, "setup"):
            saver.setup()
    else:
        saver = SqlitePersistentSaver(conn)

    inicializar_tabla_sesiones(ruta)

    if db_path is None:
        _singleton_checkpointer = saver
        _singleton_db_path = ruta

    return saver


# ============================================================================
# REGISTRO PERSISTENTE DE SESIONES (thread_id) PARA REANUDACIÓN DE TAREAS
# ============================================================================

def inicializar_tabla_sesiones(db_path: Optional[Path] = None) -> None:
    """Crea la tabla 'sesiones_facturas' en SQLite si aún no existe."""
    with _conectar_sqlite(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sesiones_facturas (
                thread_id TEXT PRIMARY KEY,
                ruta_imagen TEXT NOT NULL,
                nombre_archivo TEXT NOT NULL,
                proveedor TEXT,
                folio TEXT,
                estado TEXT NOT NULL,
                total_items INTEGER DEFAULT 0,
                items_procesados INTEGER DEFAULT 0,
                items_pendientes INTEGER DEFAULT 0,
                item_en_pausa TEXT,
                creado_en TEXT NOT NULL,
                actualizado_en TEXT NOT NULL
            )
            """
        )
        conn.commit()


def registrar_sesion(thread_id: str, ruta_imagen: str | Path, db_path: Optional[Path] = None) -> None:
    """Registra el inicio de una nueva sesión de procesamiento asociada a un thread_id."""
    inicializar_tabla_sesiones(db_path)
    ruta_obj = Path(ruta_imagen)
    ahora = datetime.now().isoformat(timespec="seconds")

    with _conectar_sqlite(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sesiones_facturas (
                thread_id, ruta_imagen, nombre_archivo, proveedor, folio,
                estado, total_items, items_procesados, items_pendientes,
                item_en_pausa, creado_en, actualizado_en
            ) VALUES (?, ?, ?, NULL, NULL, 'INICIADO', 0, 0, 0, NULL, ?, ?)
            """,
            (thread_id, str(ruta_obj), ruta_obj.name, ahora, ahora),
        )
        conn.commit()


def sincronizar_sesion_con_estado(
    thread_id: str,
    valores_estado: Dict[str, Any],
    pausada: bool = False,
    db_path: Optional[Path] = None,
) -> None:
    """
    Actualiza el registro SQLite de la sesión con los datos actuales del StateSnapshot de LangGraph.
    """
    inicializar_tabla_sesiones(db_path)
    datos_factura = valores_estado.get("datos_factura") or {}
    proveedor = datos_factura.get("proveedor")
    folio = datos_factura.get("numero_factura")

    procesados = list(valores_estado.get("items_procesados") or [])
    pendientes_cola = list(valores_estado.get("items_pendientes") or [])
    item_actual = valores_estado.get("item_actual")

    if pausada:
        estado_sesion = "PAUSADA_HITL"
        pendientes_totales = len(pendientes_cola) + (1 if item_actual else 0)
        glosa_pausa = (item_actual.get("nombre_glosa") if isinstance(item_actual, dict) else None) or "ILEGIBLE"
    else:
        estado_sesion = valores_estado.get("estado_final") or "COMPLETADO"
        pendientes_totales = len(pendientes_cola)
        glosa_pausa = None

    items_originales = list(datos_factura.get("items") or [])
    total_items = max(len(items_originales), len(procesados) + pendientes_totales)
    ahora = datetime.now().isoformat(timespec="seconds")

    ruta_imagen = valores_estado.get("ruta_imagen") or ""
    nombre_archivo = Path(ruta_imagen).name if ruta_imagen else "desconocido"

    with _conectar_sqlite(db_path) as conn:
        cursor = conn.execute(
            "SELECT creado_en FROM sesiones_facturas WHERE thread_id = ?",
            (thread_id,),
        )
        existente = cursor.fetchone()
        creado_en = existente["creado_en"] if existente else ahora

        conn.execute(
            """
            INSERT OR REPLACE INTO sesiones_facturas (
                thread_id, ruta_imagen, nombre_archivo, proveedor, folio,
                estado, total_items, items_procesados, items_pendientes,
                item_en_pausa, creado_en, actualizado_en
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                str(ruta_imagen),
                nombre_archivo,
                str(proveedor) if proveedor else None,
                str(folio) if folio else None,
                estado_sesion,
                total_items,
                len(procesados),
                pendientes_totales,
                glosa_pausa,
                creado_en,
                ahora,
            ),
        )
        conn.commit()


def listar_sesiones_pendientes(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retorna todas las sesiones pausadas o interrumpidas que pueden ser reanudadas."""
    inicializar_tabla_sesiones(db_path)
    with _conectar_sqlite(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM sesiones_facturas
            WHERE estado IN ('PAUSADA_HITL', 'EN_PROCESO', 'INICIADO')
            ORDER BY actualizado_en DESC
            """
        )
        return [dict(fila) for fila in cursor.fetchall()]


def obtener_sesion(thread_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Obtiene los metadatos registrados de una sesión específica por su thread_id."""
    inicializar_tabla_sesiones(db_path)
    with _conectar_sqlite(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM sesiones_facturas WHERE thread_id = ?
            """,
            (thread_id,),
        )
        fila = cursor.fetchone()
        return dict(fila) if fila else None
