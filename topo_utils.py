"""
topo_utils.py
Funciones de cálculo puro para el simulador topográfico.
Sin dependencias de Streamlit: solo numpy / scipy / math.
"""
import math
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import griddata


def cargar_puntos(df):
    """
    Extrae arrays numpy de X, Y, Z desde un DataFrame con columnas
    [punto, x, y, z, codigo] (insensible a mayúsculas, acepta variantes).
    """
    cols = {c.lower().strip(): c for c in df.columns}

    def buscar(*alias):
        for a in alias:
            if a in cols:
                return cols[a]
        return None

    col_x = buscar("x", "este", "easting")
    col_y = buscar("y", "norte", "northing")
    col_z = buscar("z", "cota", "elevacion", "elevación")

    if col_x is None or col_y is None or col_z is None:
        raise ValueError(
            "No se pudieron identificar las columnas X, Y, Z en el archivo. "
            "Verifica que el archivo tenga el formato: PUNTO,X,Y,Z,CODIGO"
        )

    x = df[col_x].astype(float).to_numpy()
    y = df[col_y].astype(float).to_numpy()
    z = df[col_z].astype(float).to_numpy()
    return x, y, z


def construir_triangulacion(x, y, max_dist):
    """
    Fase 3: Construye una triangulación de Delaunay y descarta los
    triángulos cuyo lado más largo supera max_dist (evita "hilos" que
    crucen vacíos sin datos).

    Devuelve:
        tri: objeto Delaunay completo (para referencia)
        simplices_validos: array (M,3) con los índices de los triángulos
                            que sí cumplen el criterio de distancia
    """
    puntos = np.column_stack([x, y])
    tri = Delaunay(puntos)

    simplices = tri.simplices
    validos = []
    for s in simplices:
        p0, p1, p2 = puntos[s[0]], puntos[s[1]], puntos[s[2]]
        d01 = np.linalg.norm(p0 - p1)
        d12 = np.linalg.norm(p1 - p2)
        d20 = np.linalg.norm(p2 - p0)
        lado_max = max(d01, d12, d20)
        if lado_max <= max_dist:
            validos.append(s)

    simplices_validos = np.array(validos) if validos else np.empty((0, 3), dtype=int)
    return tri, simplices_validos


def extraer_aristas_unicas(simplices_validos):
    """
    Fase 3 (wireframe): a partir de los triángulos válidos, extrae el
    conjunto de aristas ÚNICAS (sin duplicar la arista compartida entre
    dos triángulos vecinos). Devuelve un array (E,2) de índices de punto.

    Esto es lo que permite dibujar el "esqueleto" como líneas (huecos
    entre líneas) en vez de caras rellenas tipo superficie.
    """
    aristas = set()
    for s in simplices_validos:
        p0, p1, p2 = int(s[0]), int(s[1]), int(s[2])
        for a, b in ((p0, p1), (p1, p2), (p2, p0)):
            aristas.add((a, b) if a < b else (b, a))
    if not aristas:
        return np.empty((0, 2), dtype=int)
    return np.array(sorted(aristas), dtype=int)


def generar_malla_mde(x, y, z, simplices_validos, resolucion=80, metodo="linear"):
    """
    Fase 4: Genera una malla regular (grid) interpolando Z sobre el
    dominio triangulado válido. Las zonas fuera de los triángulos
    válidos (huecos reales del levantamiento) quedan como NaN.
    """
    xi = np.linspace(x.min(), x.max(), resolucion)
    yi = np.linspace(y.min(), y.max(), resolucion)
    XI, YI = np.meshgrid(xi, yi)

    ZI = griddata((x, y), z, (XI, YI), method=metodo)

    if simplices_validos.shape[0] > 0:
        mascara = punto_dentro_de_triangulos(XI, YI, x, y, simplices_validos)
        ZI = np.where(mascara, ZI, np.nan)

    return XI, YI, ZI


def punto_dentro_de_triangulos(XI, YI, x, y, simplices_validos):
    """
    Determina, para cada punto de la malla (XI,YI), si cae dentro de
    alguno de los triángulos válidos. Usa coordenadas baricéntricas.
    """
    puntos_grid = np.column_stack([XI.ravel(), YI.ravel()])
    mascara = np.zeros(puntos_grid.shape[0], dtype=bool)

    for s in simplices_validos:
        p0 = np.array([x[s[0]], y[s[0]]])
        p1 = np.array([x[s[1]], y[s[1]]])
        p2 = np.array([x[s[2]], y[s[2]]])

        v0 = p1 - p0
        v1 = p2 - p0
        v2 = puntos_grid - p0

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = v2 @ v0
        d21 = v2 @ v1

        denom = d00 * d11 - d01 * d01
        if denom == 0:
            continue

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        dentro = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
        mascara |= dentro

    return mascara.reshape(XI.shape)


def rellenar_bloque_solido(ZI, z_base=None):
    """
    Fase 5: Dada la superficie del terreno (ZI), calcula la cota base
    plana (la cota mínima del levantamiento, o la indicada) para poder
    extruir un bloque sólido entre esa base y la superficie del MDE.
    """
    if z_base is None:
        z_base = np.nanmin(ZI)
    return z_base


def construir_mesh_paredes(XI, YI, ZI, z_base):
    """
    Fase 5: Genera las paredes laterales de la maqueta siguiendo el
    CONTORNO REAL del terreno con datos válidos (no solo el rectángulo
    exterior de la malla). Esto cubre tanto bordes irregulares (terreno
    no rectangular) como huecos internos (zonas sin triangulación
    válida en la Fase 3): cualquier celda válida que limite con una
    celda inválida (o con el borde de la malla) recibe pared.

    Una celda de la malla (i,j) se considera "válida" solo si sus 4
    vértices tienen cota numérica (sin NaN); así la pared nace
    exactamente donde el terreno real termina.

    Devuelve arrays listos para un único trace go.Mesh3d:
        x, y, z, i, j, k
    """
    valido = ~np.isnan(ZI)
    celda_valida = valido[:-1, :-1] & valido[1:, :-1] & valido[:-1, 1:] & valido[1:, 1:]
    nfil, ncol = celda_valida.shape

    xs, ys, zs = [], [], []
    i_idx, j_idx, k_idx = [], [], []
    idx = 0

    def agregar_segmento(p1, p2):
        nonlocal idx
        i1, j1 = p1
        i2, j2 = p2
        x1, y1, z1 = XI[i1, j1], YI[i1, j1], ZI[i1, j1]
        x2, y2, z2 = XI[i2, j2], YI[i2, j2], ZI[i2, j2]
        # Quad vertical: top1 -> top2 -> base2 -> base1
        xs.extend([x1, x2, x2, x1])
        ys.extend([y1, y2, y2, y1])
        zs.extend([z1, z2, z_base, z_base])
        i_idx.extend([idx, idx])
        j_idx.extend([idx + 1, idx + 2])
        k_idx.extend([idx + 2, idx + 3])
        idx += 4

    for i in range(nfil):
        for j in range(ncol):
            if not celda_valida[i, j]:
                continue
            # borde superior de la celda (frontera con celda de arriba)
            if i == 0 or not celda_valida[i - 1, j]:
                agregar_segmento((i, j), (i, j + 1))
            # borde inferior
            if i == nfil - 1 or not celda_valida[i + 1, j]:
                agregar_segmento((i + 1, j), (i + 1, j + 1))
            # borde izquierdo
            if j == 0 or not celda_valida[i, j - 1]:
                agregar_segmento((i, j), (i + 1, j))
            # borde derecho
            if j == ncol - 1 or not celda_valida[i, j + 1]:
                agregar_segmento((i, j + 1), (i + 1, j + 1))

    if idx == 0:
        return (np.empty(0), np.empty(0), np.empty(0), [], [], [])

    return (np.array(xs), np.array(ys), np.array(zs), i_idx, j_idx, k_idx)


def pendiente_porcentaje_a_grados(pendiente_pct):
    """
    Convierte una pendiente longitudinal en porcentaje a grados
    sexagesimales exactos. Dato normativo obligatorio en la memoria
    de cálculo.

    pendiente_pct: pendiente en % (ej. 8.5 significa 8.5%)
    """
    return math.degrees(math.atan(pendiente_pct / 100.0))


def estadisticas_basicas(x, y, z):
    return {
        "n_puntos": len(z),
        "x_min": float(np.min(x)), "x_max": float(np.max(x)),
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "z_min": float(np.min(z)), "z_max": float(np.max(z)),
        "z_prom": float(np.mean(z)),
        "area_aprox_m2": float((np.max(x) - np.min(x)) * (np.max(y) - np.min(y))),
    }


# ----------------------------------------------------------------------
# FASE 7 — EJE VIAL (planta + rasante)
# ----------------------------------------------------------------------
def punto_cota_minima(x, y, z):
    """Devuelve (x, y, z) del punto de menor cota del levantamiento."""
    idx = int(np.argmin(z))
    return float(x[idx]), float(y[idx]), float(z[idx])


def punto_extremo(x, y, z, direccion):
    """
    Devuelve el punto del levantamiento más alejado en la dirección
    cardinal indicada: 'norte' (max Y), 'sur' (min Y), 'este' (max X),
    'oeste' (min X). Útil como destino automático del eje vial.
    """
    direccion = direccion.lower().strip()
    if direccion == "norte":
        idx = int(np.argmax(y))
    elif direccion == "sur":
        idx = int(np.argmin(y))
    elif direccion == "este":
        idx = int(np.argmax(x))
    elif direccion == "oeste":
        idx = int(np.argmin(x))
    else:
        raise ValueError("Dirección no válida. Usa: norte, sur, este u oeste.")
    return float(x[idx]), float(y[idx]), float(z[idx])


def generar_eje_vial(puntos_control, n_muestras=200, sigma_suavizado=3.0):
    """
    A partir de una polilínea de puntos de control (X,Y) en orden,
    genera n_muestras estaciones equiespaciadas por longitud de arco y
    aplica un filtro gaussiano 1D (scipy.ndimage.gaussian_filter1d) a
    las coordenadas X,Y para suavizar los quiebres en curvas continuas
    (sigma=0 deja el trazado anguloso original).

    Devuelve: x_eje, y_eje, abscisas (distancia acumulada en metros, PK)
    """
    pts = np.array(puntos_control, dtype=float)
    if len(pts) < 2:
        raise ValueError("Se necesitan al menos 2 puntos de control para trazar el eje.")

    deltas = np.diff(pts, axis=0)
    seg_len = np.sqrt((deltas ** 2).sum(axis=1))
    dist_acum_ctrl = np.concatenate([[0.0], np.cumsum(seg_len)])
    long_total = dist_acum_ctrl[-1]
    if long_total == 0:
        raise ValueError("Los puntos de control coinciden entre sí; no se puede trazar el eje.")

    abscisas = np.linspace(0, long_total, n_muestras)
    x_eje = np.interp(abscisas, dist_acum_ctrl, pts[:, 0])
    y_eje = np.interp(abscisas, dist_acum_ctrl, pts[:, 1])

    if sigma_suavizado > 0:
        from scipy.ndimage import gaussian_filter1d
        x_eje = gaussian_filter1d(x_eje, sigma=sigma_suavizado, mode="nearest")
        y_eje = gaussian_filter1d(y_eje, sigma=sigma_suavizado, mode="nearest")
        # El filtro gaussiano "encoge" un poco las puntas: reanclamos
        # exactamente el inicio y el fin a los puntos de control reales.
        x_eje[0], y_eje[0] = pts[0]
        x_eje[-1], y_eje[-1] = pts[-1]
        d2 = np.sqrt(np.diff(x_eje) ** 2 + np.diff(y_eje) ** 2)
        abscisas = np.concatenate([[0.0], np.cumsum(d2)])

    return x_eje, y_eje, abscisas


def interpolar_terreno_en_eje(x, y, z, x_eje, y_eje):
    """
    Interpola la cota del terreno natural bajo cada estación del eje vial.
    Usa interpolación lineal y, para estaciones que caigan fuera del
    casco convexo de puntos (extrapolación), cae a 'nearest' en vez de
    devolver NaN (el eje SIEMPRE necesita una cota de terreno).
    """
    z_eje = griddata((x, y), z, (x_eje, y_eje), method="linear")
    z_eje = np.asarray(z_eje, dtype=float)
    faltantes = np.isnan(z_eje)
    if faltantes.any():
        z_cercano = griddata((x, y), z, (x_eje[faltantes], y_eje[faltantes]), method="nearest")
        z_eje[faltantes] = z_cercano
    return z_eje


def calcular_rasante(abscisas, z_inicial, tramos):
    """
    Calcula la cota de diseño (rasante) en cada estación del eje, dados
    los tramos con pendiente constante.

    tramos: lista de dicts {"pk_ini", "pk_fin", "pendiente_pct"}
            (pendiente positiva = subiendo en el sentido de avance PK 0 -> PK final)
    """
    abscisas = np.asarray(abscisas, dtype=float)
    z_rasante = np.full_like(abscisas, np.nan, dtype=float)
    z_actual = z_inicial
    for tramo in tramos:
        pk_i, pk_f, pend = tramo["pk_ini"], tramo["pk_fin"], tramo["pendiente_pct"]
        mascara = (abscisas >= pk_i - 1e-6) & (abscisas <= pk_f + 1e-6)
        if not mascara.any():
            continue
        pk_tramo = abscisas[mascara]
        z_rasante[mascara] = z_actual + (pend / 100.0) * (pk_tramo - pk_i)
        z_actual = z_actual + (pend / 100.0) * (pk_f - pk_i)
    return z_rasante


def validar_pendientes(tramos, max_pct=12.0):
    """
    Devuelve una lista de mensajes de alerta para los tramos cuya
    pendiente (en valor absoluto) supera el máximo normativo (12% por
    defecto). Lista vacía = todo dentro de norma.
    """
    alertas = []
    for n, tramo in enumerate(tramos, start=1):
        pend = tramo["pendiente_pct"]
        if abs(pend) > max_pct:
            alertas.append(
                f"Tramo {n} (PK {tramo['pk_ini']:.0f} a {tramo['pk_fin']:.0f}): "
                f"pendiente de {pend:.1f}% supera el máximo permitido de {max_pct:.0f}%."
            )
    return alertas


# ══════════════════════════════════════════════════════════════════════
# FASE 7 — EJE VIAL
# ══════════════════════════════════════════════════════════════════════

def interpolar_z_en_puntos(xs, ys, XI, YI, ZI):
    """
    Interpola la cota del MDE en una lista arbitraria de puntos XY.
    Rellena NaN del grid con la media antes de interpolar para evitar
    que zonas de hueco rompan la consulta de puntos cercanos.
    """
    from scipy.interpolate import RegularGridInterpolator
    xi_1d = XI[0, :]
    yi_1d = YI[:, 0]
    ZI_fill = np.where(np.isnan(ZI), np.nanmean(ZI), ZI)
    rgi = RegularGridInterpolator(
        (yi_1d, xi_1d), ZI_fill,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    return rgi(np.column_stack([np.asarray(ys, dtype=float),
                                np.asarray(xs, dtype=float)]))


def suavizar_eje_gaussiano(waypoints_xy, n_puntos=500, sigma=15):
    """
    Dado un conjunto de waypoints [[x0,y0],[x1,y1],...]:
    1. Interpola con spline cúbico (≥3 puntos) o lineal (2 puntos).
    2. Aplica filtro gaussiano para redondear las esquinas.

    sigma=0  → eje de líneas rectas sin suavizado.
    sigma>0  → curvas suaves proporcionales al valor.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import CubicSpline

    wps = np.asarray(waypoints_xy, dtype=float)
    if len(wps) < 2:
        raise ValueError("Se necesitan al menos 2 waypoints para trazar el eje.")

    # Parametrización por distancia acumulada
    dists = np.concatenate([[0.0],
        np.cumsum(np.linalg.norm(np.diff(wps, axis=0), axis=1))])
    t = dists / dists[-1]
    t_fine = np.linspace(0.0, 1.0, n_puntos)

    if len(wps) >= 3:
        xf = CubicSpline(t, wps[:, 0])(t_fine)
        yf = CubicSpline(t, wps[:, 1])(t_fine)
    else:
        xf = np.interp(t_fine, t, wps[:, 0])
        yf = np.interp(t_fine, t, wps[:, 1])

    if sigma > 0:
        s = max(1.0, sigma * n_puntos / 500.0)
        xf = gaussian_filter1d(xf, sigma=s)
        yf = gaussian_filter1d(yf, sigma=s)

    return xf, yf


def calcular_distancias_acumuladas(x_eje, y_eje):
    """Distancia acumulada a lo largo del eje (PK)."""
    dx = np.diff(np.asarray(x_eje))
    dy = np.diff(np.asarray(y_eje))
    return np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])


# ══════════════════════════════════════════════════════════════════════
# FASE 8 — CÁLCULO DE VOLUMEN DE CORTE Y RELLENO (Método de Prismatoides)
# ══════════════════════════════════════════════════════════════════════

def calcular_secciones_transversales(x_eje, y_eje, dist_acum, z_terreno, z_rasante,
                                      ancho_via, XI, YI, ZI,
                                      talud_corte=1.0, talud_relleno=1.5):
    """
    Para cada estación del eje calcula la altura de corte/relleno y
    el área de la sección transversal (trapecio con taludes).

    h>0 => CORTE (excavamos), h<0 => RELLENO (rellenamos).
    Área trapecio: A = (W + talud*h) * h
    """
    dif = np.asarray(z_terreno, dtype=float) - np.asarray(z_rasante, dtype=float)
    h_corte   = np.where(dif > 0,  dif, 0.0)
    h_relleno = np.where(dif < 0, -dif, 0.0)
    W = float(ancho_via)
    area_corte   = (W + talud_corte   * h_corte)   * h_corte
    area_relleno = (W + talud_relleno * h_relleno)  * h_relleno
    return {
        "h_corte": h_corte, "h_relleno": h_relleno,
        "area_corte": area_corte, "area_relleno": area_relleno,
        "dif_cota": dif,
    }


def calcular_volumenes_acumulados(dist_acum, areas_corte, areas_relleno):
    """
    Integra áreas por el método del prismatoide:
    V_seg = (A1 + A2) / 2 * distancia_entre_estaciones
    """
    dist = np.asarray(dist_acum, dtype=float)
    ac   = np.asarray(areas_corte, dtype=float)
    ar   = np.asarray(areas_relleno, dtype=float)
    dd   = np.diff(dist)
    vc_seg = (ac[:-1] + ac[1:]) / 2.0 * dd
    vr_seg = (ar[:-1] + ar[1:]) / 2.0 * dd
    vol_corte_acum   = np.concatenate([[0.0], np.cumsum(vc_seg)])
    vol_relleno_acum = np.concatenate([[0.0], np.cumsum(vr_seg)])
    return vol_corte_acum, vol_relleno_acum


def pk_donde_se_agota_presupuesto(dist_acum, vol_corte_acum, vol_relleno_acum,
                                   presupuesto_m3):
    """
    Encuentra el PK donde el volumen total (corte+relleno) supera el presupuesto.
    Devuelve (pk_limite, idx_limite, vol_total).
    pk_limite=None si el presupuesto no se agota en el tramo.
    """
    vol_total = vol_corte_acum + vol_relleno_acum
    indices = np.where(vol_total >= presupuesto_m3)[0]
    if len(indices) == 0:
        return None, len(dist_acum) - 1, vol_total
    idx = int(indices[0])
    if idx > 0:
        v0, v1 = vol_total[idx-1], vol_total[idx]
        d0, d1 = dist_acum[idx-1], dist_acum[idx]
        frac = (presupuesto_m3 - v0) / (v1 - v0) if v1 > v0 else 0.0
        pk_lim = float(d0 + frac * (d1 - d0))
    else:
        pk_lim = float(dist_acum[idx])
    return pk_lim, idx, vol_total


def construir_mesh_carretera(x_eje, y_eje, z_rasante, dist_acum,
                              ancho_via, XI, YI, ZI,
                              h_corte, h_relleno,
                              talud_corte=1.0, talud_relleno=1.5,
                              idx_limite=None):
    """
    Genera las geometrías 3D de la carretera construida:
    - Calzada (franja plana sobre la rasante)
    - Taludes de corte (color tierra excavada)
    - Taludes de relleno (color terraplén)

    Devuelve dict con coordenadas para armar traces de Plotly.
    """
    from scipy.interpolate import RegularGridInterpolator

    n = len(x_eje) if idx_limite is None else min(idx_limite + 1, len(x_eje))
    xe = np.asarray(x_eje[:n], dtype=float)
    ye = np.asarray(y_eje[:n], dtype=float)
    zr = np.asarray(z_rasante[:n], dtype=float)
    hc = np.asarray(h_corte[:n], dtype=float)
    hr = np.asarray(h_relleno[:n], dtype=float)
    W  = float(ancho_via) / 2.0

    # Vectores perpendiculares al eje en planta
    dx = np.gradient(xe)
    dy = np.gradient(ye)
    norm = np.sqrt(dx**2 + dy**2) + 1e-12
    nx =  dy / norm
    ny = -dx / norm

    # Bordes de calzada
    bx_izq = xe + W * nx;  by_izq = ye + W * ny
    bx_der = xe - W * nx;  by_der = ye - W * ny

    # Interpolador del MDE para cotas del terreno natural
    xi_1d = XI[0, :]
    yi_1d = YI[:, 0]
    ZI_fill = np.where(np.isnan(ZI), np.nanmean(ZI), ZI)
    rgi = RegularGridInterpolator(
        (yi_1d, xi_1d), ZI_fill,
        method="linear", bounds_error=False, fill_value=np.nan
    )

    def z_ter(xs, ys):
        return rgi(np.column_stack([np.asarray(ys, float), np.asarray(xs, float)]))

    zt_izq = z_ter(bx_izq, by_izq)
    zt_der = z_ter(bx_der, by_der)
    zt_eje = z_ter(xe, ye)

    # Pie de talud de corte (alejado del eje hacia el cerro)
    tx_izq_c = bx_izq + talud_corte * hc * nx
    ty_izq_c = by_izq + talud_corte * hc * ny
    tx_der_c = bx_der - talud_corte * hc * nx
    ty_der_c = by_der - talud_corte * hc * ny

    # Pie de talud de relleno (alejado del eje hacia abajo)
    tx_izq_r = bx_izq + talud_relleno * hr * nx
    ty_izq_r = by_izq + talud_relleno * hr * ny
    tx_der_r = bx_der - talud_relleno * hr * nx
    ty_der_r = by_der - talud_relleno * hr * ny

    return {
        "calzada":  {"x_izq": bx_izq, "y_izq": by_izq, "z_izq": zr,
                     "x_der": bx_der, "y_der": by_der, "z_der": zr},
        "talud_corte": {
            "x_izq_base": bx_izq,   "y_izq_base": by_izq,   "z_izq_base": zr,
            "x_izq_top":  tx_izq_c, "y_izq_top":  ty_izq_c, "z_izq_top":  zt_izq,
            "x_der_base": bx_der,   "y_der_base": by_der,   "z_der_base": zr,
            "x_der_top":  tx_der_c, "y_der_top":  ty_der_c, "z_der_top":  zt_der,
            "mask": hc > 0.05,
        },
        "talud_relleno": {
            "x_izq_base": bx_izq,   "y_izq_base": by_izq,   "z_izq_base": zr,
            "x_izq_top":  tx_izq_r, "y_izq_top":  ty_izq_r, "z_izq_top":  zt_izq,
            "x_der_base": bx_der,   "y_der_base": by_der,   "z_der_base": zr,
            "x_der_top":  tx_der_r, "y_der_top":  ty_der_r, "z_der_top":  zt_der,
            "mask": hr > 0.05,
        },
        "xe": xe, "ye": ye, "zr": zr, "zt_eje": zt_eje, "n": n,
    }


def calcular_rasante_multitramo(dist_acum, z_inicio, pendientes_pct,
                                longitud_tramo=100.0):
    """
    Construye la rasante (cota de diseño) punto a punto aplicando
    la pendiente del tramo que corresponde según PK.

        z[i] = z[i-1] + Δdist × (pendiente% / 100)

    La rasante arranca anclada al terreno en la Estaca 0+000.
    """
    z = np.zeros(len(dist_acum))
    z[0] = z_inicio
    for i in range(1, len(dist_acum)):
        dd = dist_acum[i] - dist_acum[i - 1]
        idx = min(int(dist_acum[i] / longitud_tramo), len(pendientes_pct) - 1)
        z[i] = z[i - 1] + dd * (pendientes_pct[idx] / 100.0)
    return z