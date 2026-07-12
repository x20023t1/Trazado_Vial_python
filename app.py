"""
app.py
Simulador Computacional de Trazado Vial - ETAPA 1: Construir el Terreno
Fases:
 1. Importación de la libreta topográfica
 2. Nube de puntos 3D (gráfica con foto)
 3. Triangulación TIN (esqueleto de triángulos)
 4. Superficie MDE (arcilla sobre el esqueleto) + curvas de nivel
 5. Bloque sólido / maqueta topográfica
"""
import io
import math
import sqlite3
import tempfile
import os
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

from topo_utils import (
    cargar_puntos,
    construir_triangulacion,
    extraer_aristas_unicas,
    generar_malla_mde,
    rellenar_bloque_solido,
    construir_mesh_paredes,
    estadisticas_basicas,
    interpolar_z_en_puntos,
    suavizar_eje_gaussiano,
    calcular_distancias_acumuladas,
    calcular_rasante_multitramo,
    # Fase 8
    calcular_secciones_transversales,
    calcular_volumenes_acumulados,
    pk_donde_se_agota_presupuesto,
    construir_mesh_carretera,
)

st.set_page_config(
    page_title="Simulador de Trazado Vial",
    page_icon="🏔️",
    layout="wide",
)

COLORES_TRAMO = [
    "red", "limegreen", "cyan", "magenta",
    "orange", "yellow", "deepskyblue", "hotpink",
    "white", "coral", "aquamarine", "gold",
]

# ----------------------------------------------------------------------
# ESTADO DE LA SESIÓN
# ----------------------------------------------------------------------
if "fase_completada" not in st.session_state:
    st.session_state.fase_completada = {1: False, 2: False, 3: False,
                                         4: False, 5: False, 6: False,
                                         7: False, 8: False, 9: False, 10: False}
if "db_path" not in st.session_state:
    # Usamos un archivo temporal persistente durante la sesión
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    st.session_state.db_path = tmp.name
if "df_puntos" not in st.session_state:
    st.session_state.df_puntos = None
if "xyz" not in st.session_state:
    st.session_state.xyz = None
if "triangulacion" not in st.session_state:
    st.session_state.triangulacion = None
if "malla_mde" not in st.session_state:
    st.session_state.malla_mde = None
if "foto_nube" not in st.session_state:
    st.session_state.foto_nube = None
if "parametros_via" not in st.session_state:
    st.session_state.parametros_via = None
if "waypoints_f7" not in st.session_state:
    st.session_state.waypoints_f7 = None          # se inicializa en vista_fase7
if "eje_calculado" not in st.session_state:
    st.session_state.eje_calculado = None
if "pendientes_f7" not in st.session_state:
    st.session_state.pendientes_f7 = []
if "f10_pdf_bytes" not in st.session_state:
    st.session_state["f10_pdf_bytes"] = None
if "f10_pdf_nombre" not in st.session_state:
    st.session_state["f10_pdf_nombre"] = ""


def marcar_completada(fase):
    st.session_state.fase_completada[fase] = True


def fase_disponible(fase):
    """La fase 1 siempre está disponible; las demás requieren la anterior completa."""
    if fase == 1:
        return True
    return st.session_state.fase_completada[fase - 1]


# ----------------------------------------------------------------------
# MENÚ LATERAL — bloqueado secuencialmente
# ----------------------------------------------------------------------
st.sidebar.title("🏔️ Simulador Vial")
st.sidebar.caption("ETAPA 1 · Construcción del Terreno")
st.sidebar.divider()

opciones = {
    1: "Fase 1 · Importar libreta topográfica",
    2: "Fase 2 · Nube de puntos 3D",
    3: "Fase 3 · Triangulación TIN",
    4: "Fase 4 · Superficie MDE + curvas de nivel",
    5: "Fase 5 · Maqueta sólida 3D",
    6: "Fase 6 · Parámetros de diseño vial",
    7: "Fase 7 · Eje Vial sobre el terreno",
    8: "Fase 8 · 🚜 Meter el Tractor (Corte y Relleno)",
    9: "Fase 9 · 🗄️ Archivero de Diseños (SQLite3)",
    10: "Fase 10 · 📄 Memoria de Cálculo Legal (PDF)",
}

fase_labels = []
for n, etiqueta in opciones.items():
    disponible = fase_disponible(n)
    completada = st.session_state.fase_completada[n]
    icono = "✅" if completada else ("🔓" if disponible else "🔒")
    fase_labels.append(f"{icono} {etiqueta}")

seleccion = st.sidebar.radio(
    "Navegación",
    options=list(opciones.keys()),
    format_func=lambda n: fase_labels[n - 1],
    index=0,
    label_visibility="collapsed",
)

if not fase_disponible(seleccion):
    st.sidebar.warning("⚠️ Completa la fase anterior antes de continuar.")

st.sidebar.divider()
#f
grupos_etapas = {
    "Etapa 1 · Terreno (F1-F5)": [1, 2, 3, 4, 5],
    "Etapa 2 · Diseño Vial (F6-F8)": [6, 7, 8],
    "Etapa 3 · Registro y Reporte (F9-F10)": [9, 10],
}

for nombre_etapa, fases in grupos_etapas.items():
    completadas = sum(st.session_state.fase_completada[f] for f in fases)
    total = len(fases)
    progreso_etapa = completadas / total
    st.sidebar.progress(
        progreso_etapa,
        text=f"{nombre_etapa}: {int(progreso_etapa*100)}%",
    )

# ----------------------------------------------------------------------
# Si la fase pedida no está disponible, forzamos la última disponible
# ----------------------------------------------------------------------
if not fase_disponible(seleccion):
    for n in range(1, 6):
        if fase_disponible(n) and not st.session_state.fase_completada[n]:
            seleccion = n
            break
    else:
        seleccion = 1
    st.info("Te llevamos a la fase pendiente más reciente.")


# ========================================================================
# FASE 1 — IMPORTACIÓN
# ========================================================================
def vista_fase1():
    st.header("Fase 1 · Leer la libreta topográfica (Importación)")
    st.markdown(
        "Sube el archivo crudo del levantamiento. Formato esperado por línea "
        "(sin encabezado, separado por comas):"
    )
    st.code("PUNTO,X,Y,Z,CODIGO\n1,727613.188,9677929.526,2552.000,pt", language="text")

    archivo = st.file_uploader("Archivo de la libreta topográfica (.csv / .txt)", type=["csv", "txt"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        tiene_encabezado = st.checkbox("El archivo ya trae fila de encabezado", value=False)
    with col_b:
        separador = st.selectbox("Separador", [",", ";", "tab", "espacio"], index=0)

    sep_map = {",": ",", ";": ";", "tab": "\t", "espacio": r"\s+"}

    if archivo is not None:
        try:
            archivo.seek(0)
            if tiene_encabezado:
                df = pd.read_csv(archivo, sep=sep_map[separador], engine="python")
            else:
                df = pd.read_csv(
                    archivo, sep=sep_map[separador], engine="python", header=None,
                    names=["punto", "x", "y", "z", "codigo"],
                )
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            return

        st.success(f"Archivo leído correctamente: {len(df)} puntos detectados.")
        st.dataframe(df, use_container_width=True, height=280)

        try:
            x, y, z = cargar_puntos(df)
        except Exception as e:
            st.error(str(e))
            return

        stats = estadisticas_basicas(x, y, z)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N° de puntos", stats["n_puntos"])
        c2.metric("Cota mínima (m)", f"{stats['z_min']:.2f}")
        c3.metric("Cota máxima (m)", f"{stats['z_max']:.2f}")
        c4.metric("Desnivel total (m)", f"{stats['z_max'] - stats['z_min']:.2f}")

        st.session_state.df_puntos = df
        st.session_state.xyz = (x, y, z)

        st.divider()
        if st.button("➡️ Confirmar datos y pasar a Fase 2", type="primary"):
            marcar_completada(1)
            st.rerun()
    else:
        st.info("Esperando archivo... también puedes usar datos de ejemplo para probar el flujo.")
        if st.button("Usar datos de ejemplo (terreno simulado)"):
            rng = np.random.default_rng(42)
            n = 300
            x = rng.uniform(727400, 727900, n)
            y = rng.uniform(9677700, 9678200, n)
            xn = (x - 727400) / 500
            yn = (y - 9677700) / 500
            z = 2500 + 80*np.sin(xn*2.5 + yn*1.5) + 40*np.cos(xn*1.2) + 25*(xn-yn)**2
            z += rng.normal(0, 3, size=n)
            df = pd.DataFrame({
                "punto": np.arange(1, n+1), "x": x, "y": y, "z": z, "codigo": "pt"
            })
            st.session_state.df_puntos = df
            st.session_state.xyz = (x, y, z)
            marcar_completada(1)
            st.rerun()


# ========================================================================
# FASE 2 — NUBE DE PUNTOS 3D
# ========================================================================
def vista_fase2():
    st.header("Fase 2 · Clavar los palillos (Nube de puntos 3D)")
    st.markdown(
        "Cada punto del levantamiento se grafica en su posición real (X, Y, Z). "
        "Gira el modelo con el mouse para detectar errores de campo o confirmar la forma del terreno."
    )

    x, y, z = st.session_state.xyz

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=4,
            color=z,
            colorscale="Earth",
            colorbar=dict(title="Cota (m)"),
            opacity=0.9,
        ),
        text=[f"X: {xi:.2f}<br>Y: {yi:.2f}<br>Z: {zi:.2f}" for xi, yi, zi in zip(x, y, z)],
        hoverinfo="text",
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title="Este (X)",
            yaxis_title="Norte (Y)",
            zaxis_title="Cota (Z)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "toImageButtonOptions": {"format": "png", "filename": "nube_de_puntos_3d", "scale": 3},
        "displaylogo": False,
    })

    st.caption(
        "💡 Usa el ícono de pantalla completa del gráfico (esquina superior derecha), "
        "rota la nube hasta el ángulo deseado y usa el botón de cámara para guardar la imagen. "
        "Esa captura podrás adjuntarla luego en la memoria de cálculo en PDF."
    )

    st.divider()
    if st.button("➡️ Confirmar nube de puntos y pasar a Fase 3", type="primary"):
        marcar_completada(2)
        st.rerun()


# ========================================================================
# FASE 3 — TRIANGULACIÓN TIN
# ========================================================================
def vista_fase3():
    st.header("Fase 3 · Armar el esqueleto (Triangulación TIN)")
    st.markdown(
        "Se conectan los puntos más cercanos formando triángulos (Delaunay). "
        "Los triángulos cuyo lado más largo supera la **distancia máxima** se descartan, "
        "porque representan un salto sobre una zona sin datos reales."
    )

    x, y, z = st.session_state.xyz

    dist_default = float(np.percentile(
        np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2), 15
    )) if len(x) < 600 else 50.0

    max_dist = st.slider(
        "Distancia máxima permitida entre vértices del triángulo (m)",
        min_value=5.0, max_value=500.0,
        value=round(dist_default, 1), step=5.0,
        help="Triángulos con lados más largos que este valor se eliminan del esqueleto (huecos sin datos).",
    )

    with st.spinner("Calculando triangulación de Delaunay..."):
        tri, simplices_validos = construir_triangulacion(x, y, max_dist)

    st.session_state.triangulacion = {
        "tri": tri, "simplices_validos": simplices_validos, "max_dist": max_dist,
    }

    n_total = len(tri.simplices)
    n_validos = len(simplices_validos)
    c1, c2, c3 = st.columns(3)
    c1.metric("Triángulos totales (Delaunay)", n_total)
    c2.metric("Triángulos válidos", n_validos)
    c3.metric("Descartados (huecos)", n_total - n_validos)

    fig = go.Figure()

    if n_validos > 0:
        # Wireframe real: solo líneas conectando vértices, SIN caras
        # rellenas. Esto es lo que distingue el "esqueleto" (Fase 3) de
        # la superficie continua que se genera recién en la Fase 4.
        aristas = extraer_aristas_unicas(simplices_validos)
        xl, yl, zl = [], [], []
        for a, b in aristas:
            xl += [x[a], x[b], None]
            yl += [y[a], y[b], None]
            zl += [z[a], z[b], None]
        fig.add_trace(go.Scatter3d(
            x=xl, y=yl, z=zl,
            mode="lines",
            line=dict(color="seagreen", width=2),
            name="Esqueleto válido (aristas)",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=3, color="black"),
        name="Puntos topográficos",
    ))

    fig.update_layout(
        scene=dict(xaxis_title="Este (X)", yaxis_title="Norte (Y)", zaxis_title="Cota (Z)", aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "toImageButtonOptions": {"format": "png", "filename": "triangulacion_tin", "scale": 3},
        "displaylogo": False,
    })

    if n_validos == 0:
        st.error("Ningún triángulo cumple el criterio de distancia. Aumenta el valor del deslizador.")

    st.divider()
    if st.button("➡️ Confirmar esqueleto TIN y pasar a Fase 4", type="primary", disabled=(n_validos == 0)):
        marcar_completada(3)
        st.rerun()


# ========================================================================
# FASE 4 — SUPERFICIE MDE + CURVAS DE NIVEL
# ========================================================================
def vista_fase4():
    st.header("Fase 4 · Ponerle arcilla al esqueleto (Superficie MDE)")
    st.markdown(
        "Se interpola una superficie continua (Modelo Digital de Elevación) sobre el esqueleto "
        "triangular válido. Las zonas sin triángulos válidos quedan como huecos reales (no se inventan datos)."
    )

    x, y, z = st.session_state.xyz
    simplices_validos = st.session_state.triangulacion["simplices_validos"]

    col1, col2 = st.columns(2)
    with col1:
        resolucion = st.slider("Resolución de la malla (celdas por lado)", 30, 150, 80, step=10)
    with col2:
        n_curvas = st.slider("Número de curvas de nivel", 5, 30, 12)

    with st.spinner("Interpolando superficie MDE..."):
        XI, YI, ZI = generar_malla_mde(x, y, z, simplices_validos, resolucion=resolucion, metodo="linear")

    st.session_state.malla_mde = {"XI": XI, "YI": YI, "ZI": ZI}

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=XI, y=YI, z=ZI,
        colorscale="Earth",
        contours={
            "z": {
                "show": True,
                "start": float(np.nanmin(ZI)),
                "end": float(np.nanmax(ZI)),
                "size": (float(np.nanmax(ZI)) - float(np.nanmin(ZI))) / n_curvas,
                "color": "white",
                "width": 2,
            }
        },
        colorbar=dict(title="Cota (m)"),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        scene=dict(xaxis_title="Este (X)", yaxis_title="Norte (Y)", zaxis_title="Cota (Z)", aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "toImageButtonOptions": {"format": "png", "filename": "superficie_mde_curvas_nivel", "scale": 3},
        "displaylogo": False,
    })

    st.caption("💡 Pasa el cursor sobre la superficie para leer las coordenadas X, Y, Z exactas en cualquier punto.")

    pct_huecos = 100 * np.isnan(ZI).sum() / ZI.size
    if pct_huecos > 0:
        st.warning(f"⚠️ El {pct_huecos:.1f}% de la malla quedó como hueco (sin datos suficientes en la Fase 3).")

    st.divider()
    if st.button("➡️ Confirmar superficie MDE y pasar a Fase 5", type="primary"):
        marcar_completada(4)
        st.rerun()


# ========================================================================
# FASE 5 — BLOQUE SÓLIDO / MAQUETA
# ========================================================================
def vista_fase5():
    st.header("Fase 5 · Construir la Maqueta Topográfica (Bloque Sólido 3D)")
    st.markdown(
        "Se extruye la superficie del terreno hacia abajo hasta una base plana en la cota mínima, "
        "generando un bloque macizo tipo maqueta de arquitectura, con curvas de nivel proyectadas y rotulación de cotas."
    )

    XI, YI, ZI = st.session_state.malla_mde["XI"], st.session_state.malla_mde["YI"], st.session_state.malla_mde["ZI"]

    col1, col2 = st.columns(2)
    with col1:
        usar_base_personalizada = st.checkbox("Definir cota de base manualmente", value=False)
    with col2:
        z_base_auto = float(rellenar_bloque_solido(ZI))
        if usar_base_personalizada:
            z_base = st.number_input("Cota de la base plana (m)", value=z_base_auto, step=1.0)
        else:
            z_base = z_base_auto
            st.metric("Cota de base (automática = mínima)", f"{z_base:.2f} m")

    mostrar_pasto = st.checkbox("🌱 Aplicar textura tipo 'pasto' en la superficie", value=True)

    # Construir el bloque sólido: superficie superior + 4 paredes laterales + base
    nx, ny = ZI.shape
    fig = go.Figure()

    colorscale_superficie = [[0, "#5b3a1a"], [0.4, "#7a5c2e"], [0.7, "#6b8f3a"], [1, "#9fc25f"]] if mostrar_pasto else "Earth"

    # Superficie superior (terreno)
    fig.add_trace(go.Surface(
        x=XI, y=YI, z=ZI,
        colorscale=colorscale_superficie,
        showscale=True,
        colorbar=dict(title="Cota (m)"),
        contours={"z": {"show": True, "color": "white", "width": 1, "size": (np.nanmax(ZI)-np.nanmin(ZI))/12}},
        name="Superficie",
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
    ))

    # Base plana (color tierra)
    ZI_base = np.full_like(ZI, z_base)
    fig.add_trace(go.Surface(
        x=XI, y=YI, z=ZI_base,
        colorscale=[[0, "#4a3220"], [1, "#4a3220"]],
        showscale=False,
        opacity=0.9,
        name="Base",
    ))

    # Paredes laterales: se generan siguiendo el CONTORNO REAL del
    # terreno válido (borde irregular + cualquier hueco interno que
    # haya quedado de la Fase 3), no solo el rectángulo exterior de la
    # malla. Así no quedan zonas de la "montaña" sin pared.
    wx, wy, wz, wi, wj, wk = construir_mesh_paredes(XI, YI, ZI, z_base)
    if len(wi) > 0:
        fig.add_trace(go.Mesh3d(
            x=wx, y=wy, z=wz, i=wi, j=wj, k=wk,
            color="#7a5c3a", opacity=1.0, flatshading=True, showscale=False, name="Paredes",
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Este (X)", yaxis_title="Norte (Y)", zaxis_title="Cota (Z)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "toImageButtonOptions": {"format": "png", "filename": "maqueta_solida_3d", "scale": 3},
        "displaylogo": False,
    })

    st.caption(
        "💡 Esta es la maqueta física virtual del terreno. Captura la imagen en pantalla completa "
        "desde el ángulo que mejor muestre el relieve: esta foto se usará más adelante en la Memoria de Cálculo."
    )

    volumen_aprox = float(np.nansum(ZI - z_base)) * ((XI[0,1]-XI[0,0]) * (YI[1,0]-YI[0,0]))
    st.metric("Volumen aproximado del bloque sólido (m³)", f"{volumen_aprox:,.1f}")

    st.divider()
    if st.button("✅ Confirmar maqueta sólida — Etapa 1 completada", type="primary"):
        marcar_completada(5)
        st.rerun()

    if st.session_state.fase_completada[5]:
        st.success(
            "🎉 **Etapa 1 completada.** El terreno digital está construido: nube de puntos, "
            "esqueleto TIN, superficie MDE y maqueta sólida. "
            "La Etapa 2 (trazado de la vía: pendiente, ancho, cortes y rellenos) se desarrollará a continuación."
        )



# ========================================================================
# FASE 6 — PARÁMETROS DE DISEÑO VIAL (Etapa 2)
# ========================================================================
def vista_fase6():
    st.header("Fase 6: Geometría y Presupuesto")

    col1, col2 = st.columns(2)
    with col1:
        ancho_via = st.number_input(
            "Ancho de vía (W en metros):",
            min_value=0.01, value=10.0, step=0.5,
            help="Debe ser mayor que 0. No hay límite superior, pero se te "
                 "avisará si el valor es poco convencional (> 20 m).",
        )
        if ancho_via > 20:
            st.warning(
                f"⚠️ {ancho_via:.2f} m es un ancho de vía poco convencional "
                "(fuera del rango típico de diseño vial). ¿Seguro que es correcto?"
            )
    with col2:
        presupuesto_m3 = st.number_input(
            "Presupuesto de Tierra (Volumen Máximo M³):",
            min_value=0, value=40_000, step=1000,
            help="No hay límite superior, pero se te avisará si el valor "
                 "es exageradamente alto (> 1,000,000 m³).",
        )
        if presupuesto_m3 > 1_000_000:
            st.warning(
                f"⚠️ {presupuesto_m3:,} m³ es un presupuesto de tierra "
                "inusualmente alto. ¿Seguro que es correcto?"
            )

    if st.button("💾 Guardar Parámetros", use_container_width=True):
        if ancho_via <= 0 or presupuesto_m3 < 0:
            st.error(
                "El ancho de vía debe ser mayor que cero y el presupuesto de "
                "tierra no puede ser negativo."
            )
        else:
            if presupuesto_m3 == 0:
                st.warning(
                    "⚠️ Presupuesto en 0 m³: en la Fase 8 el tractor no podrá "
                    "avanzar (se detendrá en la Estaca 0+000) porque no hay "
                    "volumen de tierra disponible para mover."
                )
            st.session_state.parametros_via = {
                "ancho_via": ancho_via,
                "presupuesto_m3": presupuesto_m3,
            }
            marcar_completada(6)
            st.rerun()

    if st.session_state.fase_completada[6]:
        st.success("✅ Guardado. Avanza a la Fase 7.")


# ========================================================================
# FASE 7 — EJE VIAL SOBRE EL TERRENO
# ========================================================================
def vista_fase7():
    st.header("Fase 7 · Dibujar el eje vial sobre el terreno")
    st.markdown(
        "Define los **puntos de control** del camino sobre el mapa de planta. "
        "El simulador une esos puntos con una curva suavizada, calcula la longitud, "
        "te deja asignar una **pendiente distinta a cada tramo de 100 m** y "
        "dibuja la rasante en 3D sobre el terreno."
    )

    XI = st.session_state.malla_mde["XI"]
    YI = st.session_state.malla_mde["YI"]
    ZI = st.session_state.malla_mde["ZI"]

    # ── Inicializar waypoints con los extremos del levantamiento ──────
    if st.session_state.waypoints_f7 is None:
        xc = float(XI[0, XI.shape[1] // 2])
        y_ini = float(YI[int(YI.shape[0] * 0.15), 0])
        y_fin = float(YI[int(YI.shape[0] * 0.85), 0])
        st.session_state.waypoints_f7 = [
            {"x": xc, "y": y_ini},
            {"x": xc, "y": y_fin},
        ]

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN A — MAPA DE PLANTA + PUNTOS DE CONTROL
    # ════════════════════════════════════════════════════════════════
    st.subheader("🗺️ Vista de planta — lee coordenadas con el cursor")

    fig_plan = go.Figure()
    fig_plan.add_trace(go.Heatmap(
        x=XI[0, :], y=YI[:, 0], z=ZI,
        colorscale="Earth", showscale=True,
        colorbar=dict(title="Cota (m)", thickness=12, len=0.8),
        hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>",
    ))

    wps = st.session_state.waypoints_f7
    if len(wps) >= 2:
        wpx = [w["x"] for w in wps]
        wpy = [w["y"] for w in wps]
        sym = (["diamond"] + ["circle"] * (len(wps) - 2) + ["x"])
        clr = (["limegreen"] + ["yellow"] * (len(wps) - 2) + ["red"])
        fig_plan.add_trace(go.Scatter(
            x=wpx, y=wpy, mode="markers+lines",
            marker=dict(size=12, color=clr, symbol=sym,
                        line=dict(color="white", width=1.5)),
            line=dict(color="white", width=1, dash="dot"),
            name="Puntos de control",
            hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
        ))

    # Preview eje suavizado si ya fue calculado
    if st.session_state.eje_calculado is not None:
        eje = st.session_state.eje_calculado
        fig_plan.add_trace(go.Scatter(
            x=eje["x"], y=eje["y"], mode="lines",
            line=dict(color="deepskyblue", width=2.5),
            name="Eje suavizado",
        ))

    fig_plan.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Este (X)", yaxis_title="Norte (Y)",
        legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_plan, use_container_width=True)
    st.caption("💡 Pasa el cursor sobre el mapa, copia los valores X/Y que aparecen "
               "en la esquina inferior derecha y pégalos en los campos de abajo.")

    # ── Inputs de waypoints ──────────────────────────────────────────
    st.subheader("📍 Puntos de control del eje")

    # ── Acceso rápido: inicio en cota mínima → extremo cardinal ──────
    x_arr, y_arr, z_arr = st.session_state.xyz
    idx_min = int(np.argmin(z_arr))
    x_min_z, y_min_z = float(x_arr[idx_min]), float(y_arr[idx_min])

    st.caption("⚡ Inicio rápido — coloca el eje desde la **cota mínima** hacia:")
    bc1, bc2, bc3, bc4 = st.columns(4)
    destinos = {
        "⬆️ Norte": (float(x_arr[np.argmax(y_arr)]), float(y_arr[np.argmax(y_arr)])),
        "⬇️ Sur":   (float(x_arr[np.argmin(y_arr)]), float(y_arr[np.argmin(y_arr)])),
        "➡️ Este":  (float(x_arr[np.argmax(x_arr)]), float(y_arr[np.argmax(x_arr)])),
        "⬅️ Oeste": (float(x_arr[np.argmin(x_arr)]), float(y_arr[np.argmin(x_arr)])),
    }
    for col, (etiqueta, (xd, yd)) in zip([bc1, bc2, bc3, bc4], destinos.items()):
        with col:
            if st.button(etiqueta, use_container_width=True):
                st.session_state.waypoints_f7 = [
                    {"x": x_min_z, "y": y_min_z},
                    {"x": xd,      "y": yd},
                ]
                st.session_state.eje_calculado = None
                for k in list(st.session_state.keys()):
                    if k.startswith("wp_x_") or k.startswith("wp_y_"):
                        del st.session_state[k]
                st.rerun()

    st.divider()

    # ── Inicializar claves planas de session_state para los inputs ────
    # Usamos claves independientes (wp_x_0, wp_y_0, etc.) como fuente
    # de verdad. Solo las inicializamos si no existen o si el número
    # de waypoints cambió — nunca las sobreescribimos durante el render.
    wps = st.session_state.waypoints_f7
    for i, wp in enumerate(wps):
        if f"wp_x_{i}" not in st.session_state:
            st.session_state[f"wp_x_{i}"] = float(wp["x"])
        if f"wp_y_{i}" not in st.session_state:
            st.session_state[f"wp_y_{i}"] = float(wp["y"])

    to_delete = None
    for i, wp in enumerate(wps):
        n_total = len(wps)
        if i == 0:
            label = "Inicio — Estaca 0+000"
            icono = "🟢"
        elif i == n_total - 1:
            label = "Llegada — Meta"
            icono = "🔴"
        else:
            label = f"Punto intermedio {i}"
            icono = "🟡"

        c1, c2, c3 = st.columns([3, 3, 1])
        with c1:
            st.number_input(f"X · {icono} {label}",
                            key=f"wp_x_{i}", format="%.2f")
        with c2:
            st.number_input(f"Y · {icono} {label}",
                            key=f"wp_y_{i}", format="%.2f")
        with c3:
            if 0 < i < n_total - 1:
                if st.button("✕", key=f"del_wp_{i}"):
                    to_delete = i

        # Sincronizar el valor del widget de vuelta al waypoint
        wps[i]["x"] = st.session_state[f"wp_x_{i}"]
        wps[i]["y"] = st.session_state[f"wp_y_{i}"]

    if to_delete is not None:
        st.session_state.waypoints_f7.pop(to_delete)
        st.session_state.eje_calculado = None
        # Limpiar TODAS las claves planas para que se reinicialicen
        for k in list(st.session_state.keys()):
            if k.startswith("wp_x_") or k.startswith("wp_y_"):
                del st.session_state[k]
        st.rerun()

    col_add, col_sig = st.columns([2, 3])
    with col_add:
        if st.button("➕ Agregar punto intermedio"):
            n = len(st.session_state.waypoints_f7)
            x_new = (st.session_state.waypoints_f7[-2]["x"] +
                     st.session_state.waypoints_f7[-1]["x"]) / 2
            y_new = (st.session_state.waypoints_f7[-2]["y"] +
                     st.session_state.waypoints_f7[-1]["y"]) / 2
            st.session_state.waypoints_f7.insert(n - 1, {"x": x_new, "y": y_new})
            st.session_state.eje_calculado = None
            for k in list(st.session_state.keys()):
                if k.startswith("wp_x_") or k.startswith("wp_y_"):
                    del st.session_state[k]
            st.rerun()
    with col_sig:
        sigma = st.slider(
            "Suavidad del eje — σ Gaussiano",
            min_value=0, max_value=60, value=15,
            help="0 = segmentos rectos. Mayor valor = curvas más tendidas y suaves.",
        )

    if st.button("📐 Calcular eje y longitud total", use_container_width=True):
        wps_xy = [(w["x"], w["y"]) for w in st.session_state.waypoints_f7]
        try:
            xe, ye = suavizar_eje_gaussiano(wps_xy, n_puntos=500, sigma=sigma)
            dist   = calcular_distancias_acumuladas(xe, ye)
            z_ter  = interpolar_z_en_puntos(xe, ye, XI, YI, ZI)
            long_t = float(dist[-1])
            n_tr   = max(1, math.ceil(long_t / 100.0))
            st.session_state.eje_calculado = {
                "x": xe, "y": ye, "dist": dist,
                "z_terreno": z_ter,
                "longitud_total": long_t,
                "n_tramos": n_tr,
                "sigma": sigma,
            }
            # Reiniciar pendientes si cambió el número de tramos
            if len(st.session_state.pendientes_f7) != n_tr:
                st.session_state.pendientes_f7 = [10.0] * n_tr
        except Exception as e:
            st.error(f"Error al calcular el eje: {e}")
        st.rerun()

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN B — PENDIENTES + RASANTE 3D (solo si eje calculado)
    # ════════════════════════════════════════════════════════════════
    if st.session_state.eje_calculado is None:
        return

    eje   = st.session_state.eje_calculado
    long  = eje["longitud_total"]
    n_tr  = eje["n_tramos"]

    st.divider()
    st.info(f"📏 **Longitud total trazada: {long:.2f} metros.**")

    # Pendientes por tramo
    st.subheader("📉 Pendientes por tramo (cada 100 m)")

    pend_vals = list(st.session_state.pendientes_f7)
    hay_exceso = False
    cols_row = 4
    for row in range(math.ceil(n_tr / cols_row)):
        cols = st.columns(cols_row)
        for ci in range(cols_row):
            ti = row * cols_row + ci
            if ti >= n_tr:
                break
            d0 = ti * 100
            d1 = min((ti + 1) * 100, long)
            lbl = f"K0+{d0:.0f} a K0+{d1:.0f} (%)"
            with cols[ci]:
                v = st.number_input(
                    lbl, value=pend_vals[ti],
                    step=0.5, key=f"pend_{ti}", format="%.2f",
                )
                pend_vals[ti] = v
                if abs(v) > 12.0:
                    st.warning("⚠️ > 12%")
                    hay_exceso = True

    st.session_state.pendientes_f7 = pend_vals

    if hay_exceso:
        st.error("⚠️ Uno o más tramos superan la pendiente máxima normativa del 12% (MTOP Ecuador). "
                 "Revisa los tramos marcados antes de continuar.")

    if st.button("🗺️ Calcular Rasante Multitramo en 3D", use_container_width=True):
        z_ini = float(eje["z_terreno"][0]) if not np.isnan(eje["z_terreno"][0]) else float(np.nanmin(ZI))
        z_ras = calcular_rasante_multitramo(eje["dist"], z_ini, pend_vals, longitud_tramo=100.0)
        st.session_state.eje_calculado["z_rasante"] = z_ras
        st.session_state.eje_calculado["pendientes_usadas"] = list(pend_vals)
        st.rerun()

    # ── GRÁFICA 3D ──────────────────────────────────────────────────
    if "z_rasante" not in st.session_state.eje_calculado:
        return

    eje   = st.session_state.eje_calculado
    xe    = eje["x"];  ye  = eje["y"]
    dist  = eje["dist"]
    z_ter = eje["z_terreno"]
    z_ras = eje["z_rasante"]
    pend_u = eje["pendientes_usadas"]
    n_tr  = eje["n_tramos"]

    fig3d = go.Figure()

    # 1. Terreno translúcido
    fig3d.add_trace(go.Surface(
        x=XI, y=YI, z=ZI,
        colorscale=[[0, "#2d4a1e"], [0.4, "#4a7c2f"],
                    [0.7, "#6da33d"], [1.0, "#9dcc5f"]],
        opacity=0.45, showscale=False, hoverinfo="skip",
        contours={"z": {"show": True, "color": "rgba(255,255,255,0.25)",
                        "width": 1,
                        "size": (float(np.nanmax(ZI)) - float(np.nanmin(ZI))) / 10}},
        name="Terreno",
    ))

    # 2. Perfil del terreno natural a lo largo del eje (línea amarilla punteada)
    fig3d.add_trace(go.Scatter3d(
        x=xe, y=ye, z=z_ter,
        mode="lines",
        line=dict(color="yellow", width=3, dash="dash"),
        name="Terreno Natural (Z)",
    ))

    # 3. Tramos de rasante — un trace por tramo, color distinto
    for ti in range(n_tr):
        d0 = ti * 100.0
        d1 = min((ti + 1) * 100.0, long)
        mask = (dist >= d0) & (dist <= d1 + 0.5)
        color = COLORES_TRAMO[ti % len(COLORES_TRAMO)]
        pct   = pend_u[ti]
        fig3d.add_trace(go.Scatter3d(
            x=xe[mask], y=ye[mask], z=z_ras[mask],
            mode="lines",
            line=dict(color=color, width=6),
            name=f"Tramo K0+{int(d0)} ({pct:.1f}%)",
        ))

    # 4. Estaca 0+000 — diamante magenta
    fig3d.add_trace(go.Scatter3d(
        x=[xe[0]], y=[ye[0]], z=[z_ras[0]],
        mode="markers+text",
        marker=dict(size=10, color="magenta", symbol="diamond"),
        text=["0+000"], textposition="top center",
        textfont=dict(color="magenta", size=11),
        name="Estaca 0+000",
    ))

    # 5. Llegada Meta — X roja
    fig3d.add_trace(go.Scatter3d(
        x=[xe[-1]], y=[ye[-1]], z=[z_ras[-1]],
        mode="markers+text",
        marker=dict(size=12, color="red", symbol="x"),
        text=[f"Meta · {long:.0f} m"], textposition="top center",
        textfont=dict(color="red", size=11),
        name="Llegada Meta",
    ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title="Este (X)", yaxis_title="Norte (Y)", zaxis_title="Cota (Z)",
            aspectmode="data",
            bgcolor="black",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=680,
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            orientation="h", x=0, y=-0.04,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0.5)", bordercolor="gray",
        ),
        showlegend=True,
    )

    st.plotly_chart(fig3d, use_container_width=True, config={
        "toImageButtonOptions": {"format": "png", "filename": "eje_vial_3d", "scale": 3},
        "displaylogo": False,
    })

    # Métricas resumen
    c1, c2, c3 = st.columns(3)
    c1.metric("Longitud total del eje", f"{long:.1f} m")
    c2.metric("Cota rasante inicio", f"{z_ras[0]:.2f} m")
    c3.metric("Cota rasante llegada", f"{z_ras[-1]:.2f} m")

    st.divider()
    if st.button("➡️ Confirmar eje vial y pasar a Fase 8", type="primary"):
        marcar_completada(7)
        st.rerun()


# ========================================================================
# FASE 8 — METER EL TRACTOR (Corte y Relleno)
# ========================================================================
def vista_fase8():
    st.header("Fase 8 · 🚜 Meter el Tractor — Corte y Relleno")
    st.markdown(
        "El tractor avanza metro a metro por el eje vial. "
        "Donde la montaña está **más alta que la rasante** → **CORTE** (excavamos). "
        "Donde está **más baja** → **RELLENO/TERRAPLÉN** (rellenamos con tierra). "
        "El tractor se detiene cuando se agota el **Volumen Objetivo** definido en la Fase 6. "
        "Al final verás el hueco real de la carretera tallado en el modelo 3D."
    )

    # ── Datos de fases anteriores ──────────────────────────────────────
    XI  = st.session_state.malla_mde["XI"]
    YI  = st.session_state.malla_mde["YI"]
    ZI  = st.session_state.malla_mde["ZI"]
    eje = st.session_state.eje_calculado
    par = st.session_state.parametros_via

    xe   = eje["x"];  ye  = eje["y"]
    dist = eje["dist"]
    z_ter= eje["z_terreno"]
    z_ras= eje["z_rasante"]
    long = eje["longitud_total"]
    W    = par["ancho_via"]
    presupuesto = par["presupuesto_m3"]

    # ── Parámetros de taludes ──────────────────────────────────────────
    st.subheader("⚙️ Parámetros de taludes")
    col1, col2 = st.columns(2)
    with col1:
        talud_c = st.slider("Talud de CORTE  H:V", 0.25, 2.0, 1.0, 0.25,
                            help="Ej: 1.0 → por cada metro de alto, el talud sale 1 m horizontalmente")
    with col2:
        talud_r = st.slider("Talud de RELLENO H:V", 0.25, 2.5, 1.5, 0.25,
                            help="Ej: 1.5 → terraplén con pendiente más tendida")

    if st.button("🚜 ¡Calcular Corte y Relleno!", type="primary", use_container_width=True):
        with st.spinner("El tractor está trabajando... calculando secciones y volúmenes..."):
            secs = calcular_secciones_transversales(
                xe, ye, dist, z_ter, z_ras, W, XI, YI, ZI,
                talud_corte=talud_c, talud_relleno=talud_r
            )
            vc, vr = calcular_volumenes_acumulados(dist, secs["area_corte"], secs["area_relleno"])
            pk_lim, idx_lim, vol_total = pk_donde_se_agota_presupuesto(dist, vc, vr, presupuesto)
            mesh = construir_mesh_carretera(
                xe, ye, z_ras, dist, W, XI, YI, ZI,
                secs["h_corte"], secs["h_relleno"],
                talud_corte=talud_c, talud_relleno=talud_r,
                idx_limite=idx_lim,
            )
        st.session_state["f8_secs"]    = secs
        st.session_state["f8_vc"]      = vc
        st.session_state["f8_vr"]      = vr
        st.session_state["f8_vt"]      = vol_total
        st.session_state["f8_pk_lim"]  = pk_lim
        st.session_state["f8_idx_lim"] = idx_lim
        st.session_state["f8_mesh"]    = mesh
        st.session_state["f8_talud_c"] = talud_c
        st.session_state["f8_talud_r"] = talud_r
        st.rerun()

    # ── Mostrar resultados si ya se calculó ───────────────────────────
    if "f8_mesh" not in st.session_state:
        return

    secs    = st.session_state["f8_secs"]
    vc      = st.session_state["f8_vc"]
    vr      = st.session_state["f8_vr"]
    vol_tot = st.session_state["f8_vt"]
    pk_lim  = st.session_state["f8_pk_lim"]
    idx_lim = st.session_state["f8_idx_lim"]
    mesh    = st.session_state["f8_mesh"]

    # ── Métricas resumen ──────────────────────────────────────────────
    st.divider()
    st.subheader("💰 Resumen Oficial de Obra")

    long_real = float(dist[idx_lim]) if pk_lim is None else pk_lim
    vc_final  = float(vc[idx_lim])
    vr_final  = float(vr[idx_lim])

    col1, col2, col3 = st.columns(3)
    col1.metric("Longitud Construida Real",
                f"K0+{long_real:.2f} m")
    col2.metric("Volumen Corte Acumulado",
                f"{vc_final:,.2f} m³")
    col3.metric("Volumen Relleno Acumulado",
                f"{vr_final:,.2f} m³")

    if pk_lim is not None:
        st.warning(
            f"🛑 **El tractor se detuvo en K0+{pk_lim:.2f} m** porque se agotó el presupuesto "
            f"de **{presupuesto:,} m³**. Volumen total movido: {vol_tot[idx_lim]:,.1f} m³."
        )
    else:
        st.success(
            f"✅ **Carretera completa construida en toda su longitud ({long:.1f} m).** "
            f"Volumen total movido: {vol_tot[idx_lim]:,.1f} m³ "
            f"(presupuesto disponible: {presupuesto:,} m³)."
        )

    # ── VISTA 3D — El hueco de la carretera sobre el terreno ─────────
    st.divider()
    st.subheader("🏔️ Modelo 3D — Carretera tallada en el terreno")

    m   = mesh
    fig = go.Figure()

    # 1. Terreno natural (translúcido, color Earth)
    fig.add_trace(go.Surface(
        x=XI, y=YI, z=ZI,
        colorscale=[[0,"#3d2b1f"],[0.3,"#7a5c3a"],[0.6,"#a67c52"],[1.0,"#c4a882"]],
        opacity=0.55, showscale=False, hoverinfo="skip",
        contours={"z": {"show": True, "color": "rgba(255,255,255,0.18)",
                        "width": 1,
                        "size": (float(np.nanmax(ZI))-float(np.nanmin(ZI)))/10}},
        name="Terreno Natural",
    ))

    # 2. Calzada (franja gris oscuro = asfalto sobre la rasante)
    biz_x, biz_y, biz_z = m["calzada"]["x_izq"], m["calzada"]["y_izq"], m["calzada"]["z_izq"]
    bde_x, bde_y, bde_z = m["calzada"]["x_der"], m["calzada"]["y_der"], m["calzada"]["z_der"]
    n = m["n"]

    # Calzada como ribbon surface (Mesh3d de quads)
    vx = np.concatenate([biz_x, bde_x])
    vy = np.concatenate([biz_y, bde_y])
    vz = np.concatenate([biz_z, bde_z])
    ii, jj, kk = [], [], []
    for p in range(n - 1):
        # quad: p, p+1, n+p, n+p+1  → 2 triángulos
        ii += [p,     p+1]
        jj += [p+1,   n+p]
        kk += [n+p,   n+p+1]
    fig.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz, i=ii, j=jj, k=kk,
        color="#222222", opacity=1.0, flatshading=False,
        showscale=False, name="Calzada",
    ))

    # 3. Taludes de CORTE (naranja rojizo = tierra excavada)
    tc = m["talud_corte"]
    mask_c = tc["mask"]
    for lado in [("_izq", 1), ("_der", -1)]:
        suf, _ = lado
        xb = tc[f"x{suf}_base"][mask_c]; yb = tc[f"y{suf}_base"][mask_c]; zb = tc[f"z{suf}_base"][mask_c]
        xt = tc[f"x{suf}_top"][mask_c];  yt = tc[f"y{suf}_top"][mask_c];  zt = tc[f"z{suf}_top"][mask_c]
        nc = len(xb)
        if nc < 2:
            continue
        vx2 = np.concatenate([xb, xt])
        vy2 = np.concatenate([yb, yt])
        vz2 = np.concatenate([zb, zt])
        ii2, jj2, kk2 = [], [], []
        for p in range(nc - 1):
            ii2 += [p,    p+1]
            jj2 += [p+1,  nc+p]
            kk2 += [nc+p, nc+p+1]
        fig.add_trace(go.Mesh3d(
            x=vx2, y=vy2, z=vz2, i=ii2, j=jj2, k=kk2,
            color="#c1440e", opacity=0.85, flatshading=True,
            showscale=False, name="Talud Corte",
        ))

    # 4. Taludes de RELLENO (amarillo ocre = terraplén de tierra)
    tr = m["talud_relleno"]
    mask_r = tr["mask"]
    for lado in [("_izq",), ("_der",)]:
        suf = lado[0]
        xb = tr[f"x{suf}_base"][mask_r]; yb = tr[f"y{suf}_base"][mask_r]; zb = tr[f"z{suf}_base"][mask_r]
        xt = tr[f"x{suf}_top"][mask_r];  yt = tr[f"y{suf}_top"][mask_r];  zt = tr[f"z{suf}_top"][mask_r]
        nr = len(xb)
        if nr < 2:
            continue
        vx3 = np.concatenate([xb, xt])
        vy3 = np.concatenate([yb, yt])
        vz3 = np.concatenate([zb, zt])
        ii3, jj3, kk3 = [], [], []
        for p in range(nr - 1):
            ii3 += [p,    p+1]
            jj3 += [p+1,  nr+p]
            kk3 += [nr+p, nr+p+1]
        fig.add_trace(go.Mesh3d(
            x=vx3, y=vy3, z=vz3, i=ii3, j=jj3, k=kk3,
            color="#d4a843", opacity=0.85, flatshading=True,
            showscale=False, name="Talud Relleno",
        ))

    # 5. Eje de rasante construida (línea azul sólida)
    fig.add_trace(go.Scatter3d(
        x=m["xe"], y=m["ye"], z=m["zr"],
        mode="lines",
        line=dict(color="deepskyblue", width=5),
        name="Eje Construido",
    ))

    # 6. Perfil del terreno natural (línea amarilla punteada)
    fig.add_trace(go.Scatter3d(
        x=m["xe"], y=m["ye"], z=m["zt_eje"],
        mode="lines",
        line=dict(color="yellow", width=3, dash="dot"),
        name="Eje Natural (Proyectado)",
    ))

    # 7. Bordes de vía — derecho e izquierdo (líneas naranjas punteadas)
    fig.add_trace(go.Scatter3d(
        x=biz_x, y=biz_y, z=biz_z,
        mode="lines", line=dict(color="orange", width=2, dash="dash"),
        name="Derecho Vía",
    ))
    fig.add_trace(go.Scatter3d(
        x=bde_x, y=bde_y, z=bde_z,
        mode="lines", line=dict(color="orange", width=2, dash="dash"),
        name="Derecho Vía",
    ))

    # 8. Estaca 0+000
    fig.add_trace(go.Scatter3d(
        x=[m["xe"][0]], y=[m["ye"][0]], z=[m["zr"][0]],
        mode="markers+text",
        marker=dict(size=10, color="magenta", symbol="diamond"),
        text=["0+000"], textposition="top center",
        textfont=dict(color="magenta", size=11),
        name="Estaca 0+000",
    ))

    # 9. Punto de parada del tractor
    fig.add_trace(go.Scatter3d(
        x=[m["xe"][-1]], y=[m["ye"][-1]], z=[m["zr"][-1]],
        mode="markers+text",
        marker=dict(size=12, color="red", symbol="x"),
        text=[f"K0+{long_real:.0f}"],
        textposition="top center",
        textfont=dict(color="red", size=11),
        name="Llegada Meta",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Este (X)", yaxis_title="Norte (Y)", zaxis_title="Cota (Z)",
            aspectmode="data",
            bgcolor="black",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            orientation="h", x=0, y=-0.04,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0.5)", bordercolor="gray",
        ),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={
        "toImageButtonOptions": {"format": "png", "filename": "corte_relleno_3d", "scale": 3},
        "displaylogo": False,
    })

    # ── Perfil longitudinal Corte/Relleno ─────────────────────────────
    st.divider()
    st.subheader("📊 Perfil Longitudinal — Terreno vs Rasante")

    n_lim = idx_lim + 1
    dist_plot = dist[:n_lim]

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=dist_plot, y=z_ter[:n_lim],
        mode="lines", name="Terreno Natural",
        line=dict(color="#a67c52", width=2),
        fill=None,
    ))
    fig_perf.add_trace(go.Scatter(
        x=dist_plot, y=z_ras[:n_lim],
        mode="lines", name="Rasante (Diseño)",
        line=dict(color="deepskyblue", width=2.5),
        fill="tonexty",
        fillcolor="rgba(180,80,20,0.25)",
    ))
    # Área de relleno debajo de la rasante
    fig_perf.add_trace(go.Scatter(
        x=dist_plot, y=z_ter[:n_lim],
        mode="lines", showlegend=False,
        line=dict(color="#a67c52", width=0),
    ))
    fig_perf.add_trace(go.Scatter(
        x=dist_plot, y=z_ras[:n_lim],
        mode="lines", showlegend=False,
        line=dict(color="deepskyblue", width=0),
        fill="tonexty",
        fillcolor="rgba(212,168,67,0.25)",
    ))

    # Línea vertical donde se detiene el tractor
    if pk_lim is not None:
        fig_perf.add_vline(x=pk_lim, line_dash="dash", line_color="red",
                           annotation_text=f"🛑 K0+{pk_lim:.1f}", annotation_position="top right")

    fig_perf.update_layout(
        xaxis_title="Distancia acumulada (m PK)",
        yaxis_title="Cota (m.s.n.m.)",
        legend=dict(orientation="h", y=-0.2),
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # ── Curva de volumen acumulado ─────────────────────────────────────
    st.subheader("📈 Curva de Volumen Acumulado (Diagrama de Masas)")

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=dist[:n_lim], y=vc[:n_lim],
        mode="lines", name="Corte acumulado (m³)",
        line=dict(color="#c1440e", width=2),
    ))
    fig_vol.add_trace(go.Scatter(
        x=dist[:n_lim], y=vr[:n_lim],
        mode="lines", name="Relleno acumulado (m³)",
        line=dict(color="#d4a843", width=2),
    ))
    fig_vol.add_trace(go.Scatter(
        x=dist[:n_lim], y=vol_tot[:n_lim],
        mode="lines", name="TOTAL (m³)",
        line=dict(color="white", width=2.5, dash="dot"),
    ))
    fig_vol.add_hline(y=presupuesto, line_dash="dash", line_color="red",
                      annotation_text=f"Presupuesto: {presupuesto:,} m³",
                      annotation_position="bottom right")
    if pk_lim is not None:
        fig_vol.add_vline(x=pk_lim, line_dash="dash", line_color="red")

    fig_vol.update_layout(
        xaxis_title="PK (m)", yaxis_title="Volumen acumulado (m³)",
        legend=dict(orientation="h", y=-0.25),
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()
    if st.button("✅ Confirmar Fase 8 — Obra terminada 🎉", type="primary"):
        marcar_completada(8)
        st.rerun()

    if st.session_state.fase_completada[8]:
        st.success(
            "🎉 **¡Obra completada!** El tractor terminó su trabajo. "
            f"Se construyeron **K0+{long_real:.2f} m** de vía con "
            f"**{vc_final:,.1f} m³ de corte** y **{vr_final:,.1f} m³ de relleno**."
        )


# ========================================================================
# FASE 9 — ARCHIVERO DE DISEÑOS (SQLite3)
# ========================================================================
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulaciones (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre          TEXT NOT NULL,
    fecha           TEXT,
    ancho           REAL,
    presupuesto     REAL,
    talud_corte     REAL,
    talud_relleno   REAL,
    longitud_lograda REAL,
    corte           REAL,
    relleno         REAL,
    pendientes      TEXT
);
"""

def _db_conn():
    """Abre conexión a la BD de sesión."""
    con = sqlite3.connect(st.session_state.db_path)
    con.execute(DB_SCHEMA)
    con.commit()
    return con


def vista_fase9():
    st.header("Fase 9: Archivero de Diseños (SQLite3)")
    st.info("Guarda el historial de tus cálculos volumétricos para compararlos.")

    # ── Verificar que la Fase 8 tenga datos ───────────────────────────
    tiene_f8 = "f8_vc" in st.session_state and st.session_state["f8_vc"] is not None

    if tiene_f8:
        eje  = st.session_state.eje_calculado
        par  = st.session_state.parametros_via
        idx  = st.session_state.get("f8_idx_lim", len(eje["dist"]) - 1)
        vc   = st.session_state["f8_vc"]
        vr   = st.session_state["f8_vr"]
        dist = eje["dist"]
        pk_lim = st.session_state.get("f8_pk_lim", None)
        long_real = float(dist[idx]) if pk_lim is None else pk_lim

        st.subheader("💾 Guardar iteración actual")
        nombre = st.text_input("Ingresa un nombre para guardar esta simulación:",
                               value="Mi ruta")

        if st.button("💾 Guardar iteración actual en la Base de Datos",
                     use_container_width=False):
            talud_c = st.session_state.get("f8_talud_c", 1.0)
            talud_r = st.session_state.get("f8_talud_r", 1.5)
            pends   = str(eje.get("pendientes_usadas", []))
            con = _db_conn()
            con.execute(
                """INSERT INTO simulaciones
                   (nombre, fecha, ancho, presupuesto, talud_corte, talud_relleno,
                    longitud_lograda, corte, relleno, pendientes)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (nombre.strip(),
                 datetime.now().strftime("%Y-%m-%d %H:%M"),
                 par["ancho_via"],
                 par["presupuesto_m3"],
                 talud_c, talud_r,
                 round(long_real, 4),
                 round(float(vc[idx]), 4),
                 round(float(vr[idx]), 4),
                 pends)
            )
            con.commit()
            con.close()
            st.success(f"¡Iteración '{nombre.strip()}' guardada con éxito!")
    else:
        st.warning("Aún no hay datos de la Fase 8. Completa el cálculo de Corte y Relleno primero.")

    # ── Tabla de historial ─────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Historial de simulaciones guardadas")

    con = _db_conn()
    df_hist = pd.read_sql_query(
        "SELECT id, nombre, fecha, ancho, presupuesto, longitud_lograda, corte, relleno "
        "FROM simulaciones ORDER BY id DESC",
        con
    )
    con.close()

    if df_hist.empty:
        st.info("No hay simulaciones guardadas todavía.")
    else:
        st.dataframe(df_hist, use_container_width=True)

        # ── Comparativa visual ─────────────────────────────────────────
        if len(df_hist) >= 2:
            st.subheader("📊 Comparativa de rutas")
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=df_hist["nombre"], y=df_hist["longitud_lograda"],
                name="Longitud lograda (m)", marker_color="deepskyblue",
            ))
            fig_cmp.add_trace(go.Bar(
                x=df_hist["nombre"], y=df_hist["corte"],
                name="Corte (m³)", marker_color="#c1440e",
            ))
            fig_cmp.add_trace(go.Bar(
                x=df_hist["nombre"], y=df_hist["relleno"],
                name="Relleno (m³)", marker_color="#d4a843",
            ))
            fig_cmp.update_layout(
                barmode="group",
                xaxis_title="Simulación",
                yaxis_title="Valor",
                height=340,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", y=-0.25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.04)",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Eliminar registro ──────────────────────────────────────────
        with st.expander("🗑️ Eliminar una simulación"):
            ids_disponibles = df_hist["id"].tolist()
            id_del = st.selectbox("ID a eliminar", ids_disponibles)
            if st.button("Eliminar", type="secondary"):
                con2 = _db_conn()
                con2.execute("DELETE FROM simulaciones WHERE id=?", (id_del,))
                con2.commit()
                con2.close()
                st.success(f"Registro {id_del} eliminado.")
                st.rerun()

        # ── Descargar BD completa ──────────────────────────────────────
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar historial como CSV",
            data=csv_bytes,
            file_name="historial_simulaciones.csv",
            mime="text/csv",
        )

    st.divider()
    if st.button("➡️ Confirmar y pasar a Fase 10", type="primary"):
        marcar_completada(9)
        st.rerun()


# ========================================================================
# FASE 10 — MEMORIA DE CÁLCULO LEGAL (fpdf2)
# ========================================================================

# Tabla de sustitución para caracteres fuera de Latin-1 que puedan
# colarse en textos dinámicos (nombres de proyecto, etc.)
_SAFE = str.maketrans({
    "\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...",
    "\u00b3": "3",   # ³  → 3  (lo mostramos como "m3")
    "\u00b2": "2",
    "\u2192": "->",  # →
    "\u00b7": ".",   # ·
    "\u2713": "OK",  # ✓
    "\u2717": "X",   # ✗
    "\u00e9": "e", "\u00e1": "a", "\u00ed": "i",
    "\u00f3": "o", "\u00fa": "u", "\u00f1": "n",
    "\u00c9": "E", "\u00c1": "A", "\u00cd": "I",
    "\u00d3": "O", "\u00da": "U", "\u00d1": "N",
    "\u00fc": "u", "\u00e4": "a", "\u00f6": "o",
})

def _s(texto):
    """Convierte cualquier valor a string seguro para Helvetica (Latin-1)."""
    return str(texto).translate(_SAFE)


def _generar_pdf(nombre_proyecto, datos, imagen_bytes=None):
    """
    Genera la Memoria de Calculo en PDF con fpdf2 usando solo Helvetica
    (Latin-1). Todos los textos pasan por _s() para eliminar Unicode.
    Devuelve bytes del PDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    nombre_safe = _s(nombre_proyecto)

    # ── Helpers de fuente (siempre Helvetica) ─────────────────────────
    def fuente(estilo="", tam=10):
        pdf.set_font("Helvetica", estilo, tam)

    # ── Encabezado ─────────────────────────────────────────────────────
    fuente("B", 18)
    pdf.set_fill_color(30, 80, 160)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "MEMORIA DE CALCULO - TRAZADO VIAL",
             fill=True, ln=True, align="C")
    pdf.ln(2)

    fuente("B", 13)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 9, f"Proyecto: {nombre_safe}", ln=True)

    fuente("", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Fecha de emision: {datetime.now().strftime('%d/%m/%Y  %H:%M')}", ln=True)
    pdf.cell(0, 6, "Elaborado con: Simulador Computacional de Trazado Vial", ln=True)
    pdf.ln(4)
    pdf.set_draw_color(30, 80, 160)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # ── Imagen del modelo 3D ───────────────────────────────────────────
    if imagen_bytes:
        try:
            tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_img.write(imagen_bytes)
            tmp_img.close()
            fuente("B", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 7, "Vista 3D del Modelo Topografico y Carretera", ln=True)
            pdf.image(tmp_img.name, x=15, w=180)
            os.unlink(tmp_img.name)
            pdf.ln(4)
        except Exception:
            pass

    # ── Helpers de sección y fila ──────────────────────────────────────
    def seccion(titulo):
        fuente("B", 12)
        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(20, 60, 130)
        pdf.cell(0, 8, _s(titulo), fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        fuente("", 10)
        pdf.ln(1)

    def fila(etiqueta, valor, unidad=""):
        fuente("B", 10)
        pdf.cell(85, 7, _s(etiqueta), border="B")
        fuente("", 10)
        pdf.cell(0, 7, f"{_s(valor)}  {_s(unidad)}", border="B", ln=True)

    # ── 1. Parámetros de diseño ────────────────────────────────────────
    seccion("1. Parametros de Diseno Vial")
    fila("Ancho de via (W)", datos["ancho_via"], "m")
    fila("Presupuesto de tierra", f"{datos['presupuesto']:,}", "m3")
    fila("Talud de corte (H:V)", datos["talud_c"])
    fila("Talud de relleno (H:V)", datos["talud_r"])
    pdf.ln(4)

    # ── 2. Resultados del trazado ──────────────────────────────────────
    seccion("2. Resultados del Trazado")
    fila("Longitud total del eje proyectado", f"{datos['long_total']:.2f}", "m")
    fila("Longitud construida real (tractor)", f"{datos['long_real']:.2f}", "m")
    fila("Cota rasante inicio (Estaca 0+000)", f"{datos['z_ini']:.3f}", "m.s.n.m.")
    fila("Cota rasante llegada", f"{datos['z_fin']:.3f}", "m.s.n.m.")
    pdf.ln(4)

    # ── 3. Movimiento de tierras ───────────────────────────────────────
    seccion("3. Movimiento de Tierras")
    fila("Volumen de CORTE acumulado", f"{datos['vol_corte']:,.2f}", "m3")
    fila("Volumen de RELLENO acumulado", f"{datos['vol_relleno']:,.2f}", "m3")
    fila("Volumen TOTAL movido", f"{datos['vol_total']:,.2f}", "m3")
    pct_pres = 100 * datos['vol_total'] / datos['presupuesto'] if datos['presupuesto'] else 0
    fila("% del presupuesto utilizado", f"{pct_pres:.1f}", "%")
    pdf.ln(4)

    # ── 4. Pendientes por tramo ────────────────────────────────────────
    seccion("4. Pendientes por Tramo (cada 100 m)")
    pends = datos.get("pendientes", [])
    for i, p in enumerate(pends):
        d0 = i * 100
        d1 = min((i + 1) * 100, datos["long_total"])
        fila(f"  Tramo K0+{d0:.0f} -> K0+{d1:.0f}", f"{p:.2f}", "%")
    pdf.ln(4)

    # ── 5. Verificación normativa ──────────────────────────────────────
    seccion("5. Verificacion Normativa (MTOP Ecuador)")
    max_pend = max((abs(p) for p in pends), default=0)
    cumple = "CUMPLE" if max_pend <= 12.0 else "NO CUMPLE"
    fila("Pendiente maxima aplicada", f"{max_pend:.2f}",
         f"%  ->  {cumple} (limite 12%)")
    pdf.ln(4)

    # ── Firma ──────────────────────────────────────────────────────────
    pdf.ln(8)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.4)
    y_firma = pdf.get_y()
    pdf.line(30,  y_firma, 90,  y_firma)
    pdf.line(120, y_firma, 180, y_firma)
    pdf.ln(1)
    fuente("", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(70, 5, "   Responsable del diseno", align="L")
    pdf.cell(0,  5, "   Fiscalizador / Director de obra", align="L")
    pdf.ln(10)

    # ── Pie de página ──────────────────────────────────────────────────
    fuente("I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5,
             "Documento generado automaticamente - Simulador Computacional de Trazado Vial",
             align="C")

    return bytes(pdf.output())


def vista_fase10():
    st.header("Fase 10: Memoria de Cálculo Legal (fpdf2)")
    st.info("Adjunta la captura fotográfica del modelo 3D (Botón de cámara en Plotly) y emite el PDF formal.")

    # ── Datos requeridos ───────────────────────────────────────────────
    eje = st.session_state.eje_calculado
    par = st.session_state.parametros_via
    idx = st.session_state.get("f8_idx_lim", len(eje["dist"]) - 1)
    vc  = st.session_state.get("f8_vc", np.array([0.0]))
    vr  = st.session_state.get("f8_vr", np.array([0.0]))
    pk_lim = st.session_state.get("f8_pk_lim", None)
    dist   = eje["dist"]
    z_ras  = eje["z_rasante"]
    long_real = float(dist[idx]) if pk_lim is None else pk_lim

    talud_c = st.session_state.get("f8_talud_c", 1.0)
    talud_r = st.session_state.get("f8_talud_r", 1.5)

    # ── Formulario ────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        nombre_proyecto = st.text_input("Nombre del Proyecto:", value="Vía Nueva · Proyecto de Grado")
    with col2:
        st.caption("Usa el botón 📷 de la gráfica 3D (Fase 8) para exportar la imagen del modelo.")

    imagen_upload = st.file_uploader(
        "Sube la captura de pantalla de tu Maqueta (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    imagen_bytes = None
    if imagen_upload is not None:
        imagen_bytes = imagen_upload.read()
        st.image(imagen_bytes, caption="Vista previa de la maqueta", use_container_width=True)

    st.divider()

    # ── Resumen previo ─────────────────────────────────────────────────
    st.subheader("📋 Datos que irán en la Memoria")
    vol_corte  = float(vc[idx])
    vol_relleno= float(vr[idx])
    vol_total  = vol_corte + vol_relleno
    pends      = eje.get("pendientes_usadas", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Longitud construida", f"{long_real:.2f} m")
    col2.metric("Corte", f"{vol_corte:,.1f} m³")
    col3.metric("Relleno", f"{vol_relleno:,.1f} m³")
    col4.metric("Total movido", f"{vol_total:,.1f} m³")

    st.divider()
    if st.button("🖨️ Generar Memoria de Calculo PDF", type="primary", use_container_width=True):
        datos = {
            "ancho_via":   par["ancho_via"],
            "presupuesto": par["presupuesto_m3"],
            "talud_c":     talud_c,
            "talud_r":     talud_r,
            "long_total":  float(dist[-1]),
            "long_real":   long_real,
            "z_ini":       float(z_ras[0]),
            "z_fin":       float(z_ras[idx]),
            "vol_corte":   vol_corte,
            "vol_relleno": vol_relleno,
            "vol_total":   vol_total,
            "pendientes":  pends,
        }
        with st.spinner("Generando PDF..."):
            pdf_bytes = _generar_pdf(nombre_proyecto, datos, imagen_bytes)
        # Guardamos en session_state para que el botón de descarga
        # sobreviva al siguiente rerun de Streamlit
        st.session_state["f10_pdf_bytes"]  = pdf_bytes
        st.session_state["f10_pdf_nombre"] = nombre_proyecto
        marcar_completada(10)
        st.rerun()

    # Botón de descarga: siempre visible una vez generado el PDF
    if st.session_state.get("f10_pdf_bytes") is not None:
        nombre_archivo = st.session_state["f10_pdf_nombre"].replace(" ", "_").replace("/", "-")
        st.download_button(
            label="⬇️ Descargar Memoria de Calculo PDF",
            data=st.session_state["f10_pdf_bytes"],
            file_name=f"Memoria_{nombre_archivo}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        st.success("PDF listo. Haz click en el boton de arriba para descargarlo a tu carpeta de Descargas.")

    if st.session_state.fase_completada[10]:
        st.balloons()
        st.success(
            "Simulacion completa! Has recorrido las 10 fases del diseno vial: "
            "desde la libreta topografica hasta la Memoria de Calculo Legal. "
            "Tu proyecto esta listo para presentacion."
        )


# ========================================================================
# ROUTER
# ========================================================================
st.title("Simulador Computacional de Trazado Vial")
st.caption("Proyecto: Vía nueva en terreno montañoso · Etapa 1 — Construcción del Terreno")

if seleccion == 1:
    vista_fase1()
elif seleccion == 2:
    if st.session_state.xyz is None:
        st.warning("Primero completa la Fase 1.")
    else:
        vista_fase2()
elif seleccion == 3:
    if st.session_state.xyz is None:
        st.warning("Primero completa la Fase 1.")
    else:
        vista_fase3()
elif seleccion == 4:
    if st.session_state.triangulacion is None:
        st.warning("Primero completa la Fase 3.")
    else:
        vista_fase4()
elif seleccion == 5:
    if st.session_state.malla_mde is None:
        st.warning("Primero completa la Fase 4.")
    else:
        vista_fase5()
elif seleccion == 6:
    if st.session_state.malla_mde is None:
        st.warning("Primero completa la Fase 5.")
    else:
        vista_fase6()
elif seleccion == 7:
    if st.session_state.parametros_via is None:
        st.warning("Primero completa la Fase 6.")
    else:
        vista_fase7()
elif seleccion == 8:
    if st.session_state.eje_calculado is None or "z_rasante" not in st.session_state.eje_calculado:
        st.warning("Primero completa la Fase 7 (calcular la rasante).")
    else:
        vista_fase8()
elif seleccion == 9:
    if not st.session_state.fase_completada[8]:
        st.warning("Primero completa la Fase 8 (Corte y Relleno).")
    else:
        vista_fase9()
elif seleccion == 10:
    if not st.session_state.fase_completada[9]:
        st.warning("Primero completa la Fase 9 (Archivero de Diseños).")
    else:
        vista_fase10()