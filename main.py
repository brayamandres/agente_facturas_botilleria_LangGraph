"""
Punto de Entrada Principal (CLI) - Agente de Facturas para Botillería
Permite interactuar con el agente basado en LangGraph con persistencia real en SQLite:
- Procesar facturas nuevas y pausar en cualquier intervención humana (interrupt).
- Reanudar facturas pendientes tras cerrar y volver a abrir el programa (Command(resume=...)).
- Probar control de duplicados, inspeccionar inventario y consultar auditoría en Firestore.
"""
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Asegurar que el directorio raíz esté en el PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langgraph.types import Command

from src.config import DATA_DIR, CHECKPOINT_DB_PATH
from src.agent import (
    obtener_grafo_agente,
    construir_payload_intervencion,
    solicitar_decision_humana_consola,
)
from src.services.checkpointer import (
    registrar_sesion,
    sincronizar_sesion_con_estado,
    listar_sesiones_pendientes,
)
from src.services.firebase import get_productos_collection, get_historial_collection
from src.services.seed import sembrar_catalogo_inicial


def imprimir_banner():
    """Muestra el banner de presentación del agente."""
    print("=" * 75)
    print("      SISTEMA AGÉNTICO DE HOMOLOGACIÓN DE FACTURAS E INVENTARIO")
    print("             Botillería - LangGraph, SQLite & Firebase")
    print("=" * 75)


def listar_facturas_disponibles():
    """Lista las imágenes de facturas existentes en la carpeta data/."""
    if not DATA_DIR.exists():
        return []
    extensiones = (".png", ".jpg", ".jpeg")
    return sorted(
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(extensiones) and os.path.isfile(DATA_DIR / f)
    )


def _extraer_payload_interrupt(snapshot: Any) -> Dict[str, Any]:
    """
    Extrae el payload estructurado enviado por interrupt() desde el StateSnapshot de LangGraph.
    Incluye respaldo directo desde snapshot.values para máxima compatibilidad entre versiones.
    """
    interrupts_top = getattr(snapshot, "interrupts", None)
    if interrupts_top:
        val = getattr(interrupts_top[0], "value", None)
        if isinstance(val, dict):
            return val

    for tarea in getattr(snapshot, "tasks", ()) or ():
        intr_tarea = getattr(tarea, "interrupts", None)
        if intr_tarea:
            val = getattr(intr_tarea[0], "value", None)
            if isinstance(val, dict):
                return val

    return construir_payload_intervencion(snapshot.values or {})


def _tiene_interrupt_activo(snapshot: Any) -> bool:
    """Determina si el grafo está detenido específicamente esperando un Command(resume=...)."""
    if getattr(snapshot, "interrupts", None):
        return True
    for tarea in getattr(snapshot, "tasks", ()) or ():
        if getattr(tarea, "interrupts", None):
            return True
    siguientes = tuple(getattr(snapshot, "next", ()) or ())
    return "intervencion_humana" in siguientes


def _imprimir_aviso_pausa(thread_id: str, valores: Dict[str, Any]) -> None:
    """Muestra confirmación detallada cuando el operador pausa una factura para retomarla después."""
    procesados = list(valores.get("items_procesados") or [])
    pendientes_cola = list(valores.get("items_pendientes") or [])
    item_actual = valores.get("item_actual")
    pendientes_totales = len(pendientes_cola) + (1 if item_actual else 0)

    print("\n" + "*" * 75)
    print(" ⏸️  FACTURA PAUSADA Y PERSISTIDA EN CHECKPOINTER SQLITE")
    print("*" * 75)
    print(f"  • Thread ID de sesión:   {thread_id}")
    print(f"  • Archivo SQLite:        {CHECKPOINT_DB_PATH}")
    print(f"  • Ítems ya aplicados:    {len(procesados)} (guardados en Firestore, no se repetirán)")
    print(f"  • Ítems pendientes:      {pendientes_totales} (1 en revisión actual + {len(pendientes_cola)} en cola)")
    print("-" * 75)
    print("  Puedes cerrar completamente el programa (Opción 7) y retomarla más tarde")
    print("  seleccionando la Opción 2: 'Reanudar factura pendiente'.")
    print("*" * 75 + "\n")


def _conducir_ejecucion_interactiva(grafo: Any, thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Gestiona el ciclo de pausa/reanudación de LangGraph sobre un thread_id activo.
    Si el grafo se detiene en interrupt(), solicita la decisión humana o permite pausar
    la sesión en disco; si continúa, reanuda con Command(resume=...).
    """
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        snapshot = grafo.get_state(config)
        valores = snapshot.values or {}
        nodos_siguientes = tuple(snapshot.next or ())

        # Si no quedan nodos siguientes, la ejecución del grafo finalizó por completo
        if not nodos_siguientes:
            sincronizar_sesion_con_estado(thread_id, valores, pausada=False)
            imprimir_resumen_final(valores, thread_id=thread_id)
            return valores

        # Si hay nodos pendientes pero no es un interrupt() (ej. corte abrupto entre nodos), continuar desde checkpoint
        if not _tiene_interrupt_activo(snapshot):
            sincronizar_sesion_con_estado(thread_id, valores, pausada=False)
            grafo.invoke(None, config=config)
            continue

        # El grafo se encuentra pausado en interrupt() dentro de node_intervencion_humana
        sincronizar_sesion_con_estado(thread_id, valores, pausada=True)
        payload = _extraer_payload_interrupt(snapshot)
        decision_humana = solicitar_decision_humana_consola(payload, permitir_pausa=True)

        if decision_humana.get("tipo_accion") == "PAUSAR_SESION":
            sincronizar_sesion_con_estado(thread_id, valores, pausada=True)
            _imprimir_aviso_pausa(thread_id, valores)
            return None

        # Reanudar ejecución nativa de LangGraph desde el checkpoint SQLite
        grafo.invoke(Command(resume=decision_humana), config=config)


def ejecutar_agente_factura(ruta_imagen: Path, grafo: Optional[Any] = None):
    """
    Inicia una nueva sesión de procesamiento de factura con un thread_id persistido en SQLite.
    """
    grafo_activo = grafo or obtener_grafo_agente()
    thread_id = f"sesion-factura-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "#" * 75)
    print(" INICIANDO FLUJO DEL AGENTE CON LANGGRAPH (PERSISTENCIA SQLITE)")
    print(f" Archivo a procesar: {ruta_imagen.name}")
    print(f" Thread ID asignado: {thread_id}")
    print("#" * 75)

    estado_inicial = {
        "ruta_imagen": str(ruta_imagen),
        "datos_factura": {},
        "es_duplicada": False,
        "info_duplicada": None,
        "items_pendientes": [],
        "item_actual": None,
        "candidatos_actuales": [],
        "decision_actual": None,
        "alerta_precio": None,
        "items_procesados": [],
        "requiere_intervencion": False,
        "datos_intervencion_humana": None,
        "auditoria_id": None,
        "mensajes_bitacora": [f"Sesión iniciada con thread_id: '{thread_id}'."],
        "estado_final": "INICIADO",
    }

    try:
        registrar_sesion(thread_id, ruta_imagen)
        grafo_activo.invoke(estado_inicial, config=config)
        _conducir_ejecucion_interactiva(grafo_activo, thread_id)
    except KeyboardInterrupt:
        snapshot = grafo_activo.get_state(config)
        if snapshot and snapshot.values:
            sincronizar_sesion_con_estado(thread_id, snapshot.values, pausada=True)
            _imprimir_aviso_pausa(thread_id, snapshot.values)
        else:
            print(f"\n⏸️ Ejecución interrumpida. Sesión '{thread_id}' registrada en SQLite.")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución del agente: {e}")
        print("Verifica que las credenciales en .env y los servicios de Firebase/Gemini estén correctamente configurados.")


def mostrar_contexto_recuperado(thread_id: str, valores: Dict[str, Any]) -> None:
    """
    Despliega en detalle el estado recuperado desde el checkpointer SQLite antes de reanudar:
    ítems ya procesados, decisiones previas tomadas por el agente e ítems pendientes.
    """
    datos = valores.get("datos_factura") or {}
    procesados = list(valores.get("items_procesados") or [])
    pendientes_cola = list(valores.get("items_pendientes") or [])
    item_actual = valores.get("item_actual")
    bitacora = list(valores.get("mensajes_bitacora") or [])

    print("\n" + "=" * 75)
    print("         ESTADO RECUPERADO DESDE CHECKPOINTER PERSISTENTE (SQLITE)")
    print("=" * 75)
    print(f" Thread ID:          {thread_id}")
    print(f" Archivo Factura:    {Path(valores.get('ruta_imagen', 'N/A')).name}")
    print(f" Proveedor:          {datos.get('proveedor') or 'N/A'}")
    print(f" Folio Factura:      {datos.get('numero_factura') or 'N/A'}")
    print("-" * 75)

    # 1. Ítems ya procesados previamente (NO se volverán a ejecutar)
    print(f" [1] ÍTEMS YA PROCESADOS PREVIAMENTE ({len(procesados)}) — No se repetirán:")
    if procesados:
        for idx, it in enumerate(procesados, start=1):
            glosa = (it.get("glosa_original") or "N/A")[:28]
            sku = (it.get("sku_homologado") or "SIN_SKU")[:15]
            cant = it.get("cantidad", 0)
            est = it.get("estado", "N/A")
            print(f"     {idx:>2}. {glosa:<28} | SKU: {sku:<15} | Cant: {cant:<4} | {est}")
    else:
        print("     (Ningún ítem aplicado aún; la pausa ocurrió en el primer ítem requerido)")

    # 2. Decisiones previas registradas en la bitácora del agente
    print("-" * 75)
    ultimos_eventos = bitacora[-5:]
    print(f" [2] ÚLTIMAS DECISIONES REGISTRADAS EN BITÁCORA ({len(ultimos_eventos)} de {len(bitacora)}):")
    for ev in ultimos_eventos:
        print(f"     • {ev}")

    # 3. Ítems pendientes por procesar
    print("-" * 75)
    total_pendientes = len(pendientes_cola) + (1 if item_actual else 0)
    print(f" [3] ÍTEMS PENDIENTES POR PROCESAR ({total_pendientes}):")
    if isinstance(item_actual, dict):
        print(
            f"     -> [EN PAUSA HITL] '{item_actual.get('nombre_glosa') or 'ILEGIBLE'}' "
            f"(Cant: {item_actual.get('cantidad')}, Precio: ${item_actual.get('precio_unitario')})"
        )
    for idx_p, pend in enumerate(pendientes_cola[:5], start=1):
        print(f"     -> [En cola +{idx_p}]   '{pend.get('nombre_glosa') or 'ILEGIBLE'}'")
    if len(pendientes_cola) > 5:
        print(f"     ... y {len(pendientes_cola) - 5} ítem(s) adicional(es) en cola.")
    print("=" * 75)


def reanudar_factura_pendiente(grafo: Optional[Any] = None) -> Optional[Path]:
    """
    Lista las sesiones con facturas pausadas en SQLite, recupera su estado completo
    (ítems procesados, decisiones previas e ítems pendientes) y retoma la ejecución
    con Command(resume=...) sin repetir OCR ni actualizaciones previas.
    """
    grafo_activo = grafo or obtener_grafo_agente()
    sesiones = listar_sesiones_pendientes()

    if not sesiones:
        print("\n✅ No existen facturas pendientes o pausadas en el checkpointer SQLite.")
        return None

    print("\n" + "=" * 75)
    print("            SESIONES PENDIENTES EN CHECKPOINTER SQLITE")
    print("=" * 75)
    for idx, ses in enumerate(sesiones, start=1):
        tid = ses["thread_id"]
        arch = ses["nombre_archivo"]
        folio = ses.get("folio") or "S/F"
        prov = (ses.get("proveedor") or "Desconocido")[:18]
        proc = ses.get("items_procesados", 0)
        tot = ses.get("total_items", 0)
        pausa_en = (ses.get("item_en_pausa") or "-")[:22]
        fecha_act = ses.get("actualizado_en", "")
        print(
            f"  [{idx}] {tid} | Archivo: {arch} | Folio: {folio} | Prov: {prov}\n"
            f"      Progreso: {proc}/{tot} ítems aplicados | En pausa: '{pausa_en}' | Últ. act: {fecha_act}"
        )
    print("-" * 75)

    seleccion = input(
        f"Selecciona una sesión (1-{len(sesiones)}), escribe el thread_id, "
        f"Enter para [1] o 'c' para cancelar: "
    ).strip()

    if seleccion.lower() in ["c", "cancelar"]:
        return None

    thread_id_elegido = sesiones[0]["thread_id"]
    ruta_factura = Path(sesiones[0]["ruta_imagen"])

    if seleccion.isdigit() and 1 <= int(seleccion) <= len(sesiones):
        elegida = sesiones[int(seleccion) - 1]
        thread_id_elegido = elegida["thread_id"]
        ruta_factura = Path(elegida["ruta_imagen"])
    elif seleccion:
        coincidencia = next((s for s in sesiones if s["thread_id"] == seleccion), None)
        thread_id_elegido = seleccion
        if coincidencia:
            ruta_factura = Path(coincidencia["ruta_imagen"])

    config = {"configurable": {"thread_id": thread_id_elegido}}
    snapshot = grafo_activo.get_state(config)

    if not snapshot or not snapshot.values:
        print(f"\n⚠️ No se encontró un checkpoint válido en SQLite para thread_id='{thread_id_elegido}'.")
        return None

    mostrar_contexto_recuperado(thread_id_elegido, snapshot.values)

    try:
        _conducir_ejecucion_interactiva(grafo_activo, thread_id_elegido)
        return ruta_factura
    except KeyboardInterrupt:
        snap_post = grafo_activo.get_state(config)
        if snap_post and snap_post.values:
            sincronizar_sesion_con_estado(thread_id_elegido, snap_post.values, pausada=True)
            _imprimir_aviso_pausa(thread_id_elegido, snap_post.values)
        return ruta_factura
    except Exception as e:
        print(f"\n❌ Error al reanudar la sesión '{thread_id_elegido}': {e}")
        return None


def imprimir_resumen_final(estado, thread_id: Optional[str] = None):
    """Muestra un resumen estructurado al finalizar el procesamiento."""
    print("\n" + "=" * 75)
    print("                    REPORTE CONSOLIDADO DEL AGENTE")
    print("=" * 75)

    datos = estado.get("datos_factura") or {}
    if thread_id:
        print(f" Thread ID:          {thread_id}")
    print(f" Proveedor:          {datos.get('proveedor') or 'N/A'}")
    print(f" Folio Factura:      {datos.get('numero_factura') or 'N/A'}")
    print(f" Estado Final:       {estado.get('estado_final')}")
    print(f" Auditoría ID:       {estado.get('auditoria_id') or 'No generado'}")

    items = estado.get("items_procesados", [])
    print(f" Ítems procesados:   {len(items)}")
    print("-" * 75)

    if items:
        print(f"{'GLOSA FACTURA':<32} | {'SKU ASIGNADO':<16} | {'CANT':<5} | {'ESTADO'}")
        print("-" * 75)
        for item in items:
            glosa = (item.get("glosa_original") or "N/A")[:30]
            sku = (item.get("sku_homologado") or "SIN_SKU")[:15]
            cant = item.get("cantidad", 0)
            est = item.get("estado", "N/A")
            print(f"{glosa:<32} | {sku:<16} | {cant:<5} | {est}")
    elif estado.get("es_duplicada"):
        print(" (No se aplicaron cambios en inventario debido a control de duplicidad)")
    else:
        print(" (No se procesaron ítems)")

    print("=" * 75 + "\n")


def ver_catalogo_firestore():
    """Consulta y lista los primeros productos registrados en Firestore."""
    try:
        coleccion = get_productos_collection()
        docs = list(coleccion.limit(15).stream())

        print("\n" + "=" * 75)
        print(f" CATÁLOGO ACTUAL EN FIRESTORE (Primeros {len(docs)} productos)")
        print("=" * 75)
        print(f"{'SKU':<18} | {'NOMBRE':<32} | {'STOCK':<6} | {'COSTO':<8} | {'VENTA'}")
        print("-" * 75)

        for doc in docs:
            d = doc.to_dict()
            sku = doc.id[:17]
            nom = (d.get("nombre") or "S/N")[:30]
            stock = d.get("stock_actual", 0)
            costo = f"${d.get('precio_costo', 0):,.0f}"
            venta = f"${d.get('precio_venta', 0):,.0f}"
            print(f"{sku:<18} | {nom:<32} | {stock:<6} | {costo:<8} | {venta}")

        print("=" * 75 + "\n")
    except Exception as e:
        print(f"\n❌ Error al consultar Firestore: {e}")


def ver_historial_auditoria():
    """Consulta los registros de la colección historial_facturas (Memoria a largo plazo)."""
    try:
        coleccion = get_historial_collection()
        docs = list(coleccion.order_by("fecha_procesamiento", direction="DESCENDING").limit(10).stream())

        print("\n" + "=" * 75)
        print(" MEMORIA DE LARGO PLAZO - HISTORIAL DE FACTURAS PROCESADAS")
        print("=" * 75)

        if not docs:
            print(" No hay registros de facturas procesadas en el historial.")
        else:
            print(f"{'ID AUDITORÍA':<18} | {'FOLIO':<10} | {'PROVEEDOR':<18} | {'ITEMS':<5} | {'ESTADO'}")
            print("-" * 75)
            for doc in docs:
                d = doc.to_dict()
                aud_id = doc.id[:17]
                folio = str(d.get("folio", "N/A"))[:9]
                prov = str(d.get("proveedor", "N/A"))[:17]
                total = d.get("total_items", 0)
                estado_doc = str(d.get("estado", "COMPLETADO"))
                print(f"{aud_id:<18} | {folio:<10} | {prov:<18} | {total:<5} | {estado_doc}")

        print("=" * 75 + "\n")
    except Exception as e:
        print(f"\n❌ Error al consultar historial de auditoría: {e}")


def menu_principal():
    """Despliega el menú interactivo reutilizando una única instancia compilada del grafo con SQLite."""
    grafo = obtener_grafo_agente()
    ultima_factura_procesada: Optional[Path] = None

    while True:
        imprimir_banner()
        sesiones_pendientes = listar_sesiones_pendientes()
        aviso_pendientes = (
            f"{len(sesiones_pendientes)} sesión(es) pausada(s) en SQLite"
            if sesiones_pendientes
            else "Sin facturas pausadas"
        )
        etiqueta_dup = (
            f"Re-procesar última factura: '{ultima_factura_procesada.name}'"
            if ultima_factura_procesada
            else "Re-procesar misma factura"
        )

        print("Selecciona una opción:")
        print("  1. Procesar una factura nueva (Ejecutar Agente LangGraph)")
        print(f"  2. Reanudar factura pendiente ({aviso_pendientes})")
        print(f"  3. Probar control de duplicados ({etiqueta_dup})")
        print("  4. Visualizar inventario en Firestore")
        print("  5. Consultar historial de auditoría (Memoria de largo plazo)")
        print("  6. Sembrar / Indexar catálogo inicial con Embeddings en Firestore")
        print("  7. Salir")
        print("-" * 75)

        opcion = input("Ingresa tu opción (1-7): ").strip()

        if opcion == "1":
            facturas = listar_facturas_disponibles()
            if not facturas:
                print(f"\n⚠️ No se encontraron facturas en '{DATA_DIR}'. Agrega una imagen en formato .png o .jpg.")
                input("\nPresiona Enter para continuar...")
                continue

            print("\nFacturas disponibles en /data:")
            for idx, f in enumerate(facturas, start=1):
                print(f"  [{idx}] {f}")

            seleccion = input(f"\nSelecciona una factura (1-{len(facturas)}) o presiona Enter para usar '{facturas[0]}': ").strip()

            if seleccion.isdigit() and 1 <= int(seleccion) <= len(facturas):
                ruta_seleccionada = DATA_DIR / facturas[int(seleccion) - 1]
            else:
                ruta_seleccionada = DATA_DIR / facturas[0]

            ultima_factura_procesada = ruta_seleccionada
            ejecutar_agente_factura(ruta_seleccionada, grafo=grafo)
            input("\nPresiona Enter para volver al menú...")

        elif opcion == "2":
            ruta_reanudada = reanudar_factura_pendiente(grafo=grafo)
            if ruta_reanudada and ruta_reanudada.exists():
                ultima_factura_procesada = ruta_reanudada
            input("\nPresiona Enter para volver al menú...")

        elif opcion == "3":
            facturas = listar_facturas_disponibles()
            if not facturas:
                print(f"\n⚠️ No hay facturas en '{DATA_DIR}'.")
            elif ultima_factura_procesada and ultima_factura_procesada.exists():
                print(f"\nRe-procesando última factura procesada ('{ultima_factura_procesada.name}') para demostrar el bloqueo adaptativo ante duplicados...")
                ejecutar_agente_factura(ultima_factura_procesada, grafo=grafo)
            else:
                print("\nNo se ha procesado ninguna factura en esta sesión aún. Selecciona cuál deseas probar:")
                for idx, f in enumerate(facturas, start=1):
                    print(f"  [{idx}] {f}")
                seleccion = input(f"\nSelecciona una factura (1-{len(facturas)}) o presiona Enter para usar '{facturas[0]}': ").strip()
                if seleccion.isdigit() and 1 <= int(seleccion) <= len(facturas):
                    ruta_seleccionada = DATA_DIR / facturas[int(seleccion) - 1]
                else:
                    ruta_seleccionada = DATA_DIR / facturas[0]

                ultima_factura_procesada = ruta_seleccionada
                print(f"\nRe-procesando '{ruta_seleccionada.name}' para demostrar el bloqueo adaptativo ante duplicados...")
                ejecutar_agente_factura(ruta_seleccionada, grafo=grafo)

            input("\nPresiona Enter para volver al menú...")

        elif opcion == "4":
            ver_catalogo_firestore()
            input("Presiona Enter para volver al menú...")

        elif opcion == "5":
            ver_historial_auditoria()
            input("Presiona Enter para volver al menú...")

        elif opcion == "6":
            try:
                sembrar_catalogo_inicial()
            except Exception as e:
                print(f"\n❌ Error al sembrar catálogo: {e}")
            input("Presiona Enter para volver al menú...")

        elif opcion == "7":
            print("\n¡Gracias por utilizar el Agente de Facturas para Botillería! Hasta luego.\n")
            break
        else:
            print("\nOpción no válida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    menu_principal()
