"""
Punto de Entrada Principal (CLI) - Agente de Facturas para Botillería
Permite interactuar con el agente basado en LangGraph para procesar facturas,
inspeccionar inventario y consultar la memoria histórica de auditoría.
"""
import os
import sys
import uuid
from pathlib import Path

# Asegurar que el directorio raíz esté en el PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import DATA_DIR
from src.agent import crear_grafo_agente
from src.services.firebase import get_productos_collection, get_historial_collection
from src.services.seed import sembrar_catalogo_inicial


def imprimir_banner():
    """Muestra el banner de presentación del agente."""
    print("=" * 75)
    print("      SISTEMA AGÉNTICO DE HOMOLOGACIÓN DE FACTURAS E INVENTARIO")
    print("                  Botillería - LangGraph & Firebase")
    print("=" * 75)


def listar_facturas_disponibles():
    """Lista las imágenes de facturas existentes en la carpeta data/."""
    if not DATA_DIR.exists():
        return []
    extensiones = (".png", ".jpg", ".jpeg")
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(extensiones) and os.path.isfile(DATA_DIR / f)]


def ejecutar_agente_factura(ruta_imagen):
    """
    Ejecuta el grafo compilado de LangGraph con un thread_id único para la sesión.
    """
    print("\n" + "#" * 75)
    print(f" INICIANDO FLUJO DEL AGENTE CON LANGGRAPH")
    print(f" Archivo a procesar: {ruta_imagen.name}")
    print("#" * 75)

    # Generar un thread_id único para la memoria de corto plazo de esta sesión
    thread_id = f"sesion-factura-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Estado inicial del grafo
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
        "estado_final": "INICIADO"
    }

    try:
        grafo = crear_grafo_agente()
        # Invocar el grafo con su estado inicial y configuración de memoria
        estado_final = grafo.invoke(estado_inicial, config=config)

        # Mostrar reporte final consolidado
        imprimir_resumen_final(estado_final)

    except Exception as e:
        print(f"\n❌ Error durante la ejecución del agente: {e}")
        print("Verifica que las credenciales en .env y los servicios de Firebase/Gemini estén correctamente configurados.")


def imprimir_resumen_final(estado):
    """Muestra un resumen estructurado al finalizar el procesamiento."""
    print("\n" + "=" * 75)
    print("                    REPORTE CONSOLIDADO DEL AGENTE")
    print("=" * 75)

    datos = estado.get("datos_factura") or {}
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
            print(f"{'ID AUDITORÍA':<22} | {'FOLIO':<12} | {'PROVEEDOR':<22} | {'ITEMS'}")
            print("-" * 75)
            for doc in docs:
                d = doc.to_dict()
                aud_id = doc.id[:20]
                folio = str(d.get("folio", "N/A"))[:11]
                prov = str(d.get("proveedor", "N/A"))[:20]
                total = d.get("total_items", 0)
                print(f"{aud_id:<22} | {folio:<12} | {prov:<22} | {total}")

        print("=" * 75 + "\n")
    except Exception as e:
        print(f"\n❌ Error al consultar historial de auditoría: {e}")


def menu_principal():
    """Despliega el menú interactivo para el usuario y evaluadores."""
    ultima_factura_procesada = None

    while True:
        imprimir_banner()
        etiqueta_dup = (
            f"Re-procesar última factura: '{ultima_factura_procesada.name}'"
            if ultima_factura_procesada
            else "Re-procesar misma factura"
        )
        print("Selecciona una opción:")
        print("  1. Procesar una factura (Ejecutar Agente LangGraph)")
        print(f"  2. Probar control de duplicados ({etiqueta_dup})")
        print("  3. Visualizar inventario en Firestore")
        print("  4. Consultar historial de auditoría (Memoria de largo plazo)")
        print("  5. Sembrar / Indexar catálogo inicial con Embeddings en Firestore")
        print("  6. Salir")
        print("-" * 75)

        opcion = input("Ingresa tu opción (1-6): ").strip()

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
            ejecutar_agente_factura(ruta_seleccionada)
            input("\nPresiona Enter para volver al menú...")

        elif opcion == "2":
            facturas = listar_facturas_disponibles()
            if not facturas:
                print(f"\n⚠️ No hay facturas en '{DATA_DIR}'.")
            elif ultima_factura_procesada and ultima_factura_procesada.exists():
                print(f"\nRe-procesando última factura procesada ('{ultima_factura_procesada.name}') para demostrar el bloqueo adaptativo ante duplicados...")
                ejecutar_agente_factura(ultima_factura_procesada)
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
                ejecutar_agente_factura(ruta_seleccionada)

            input("\nPresiona Enter para volver al menú...")

        elif opcion == "3":
            ver_catalogo_firestore()
            input("Presiona Enter para volver al menú...")

        elif opcion == "4":
            ver_historial_auditoria()
            input("Presiona Enter para volver al menú...")

        elif opcion == "5":
            try:
                sembrar_catalogo_inicial()
            except Exception as e:
                print(f"\n❌ Error al sembrar catálogo: {e}")
            input("Presiona Enter para volver al menú...")

        elif opcion == "6":
            print("\n¡Gracias por utilizar el Agente de Facturas para Botillería! Hasta luego.\n")
            break
        else:
            print("\nOpción no válida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    menu_principal()
