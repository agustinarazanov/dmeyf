"""Estilo de los graficos de los videos.

Los tokens salen de la paleta de referencia de la skill `dataviz`, validada con
su propio script: los cuatro primeros slots categoricos pasan las seis pruebas
sobre la superficie clara (peor par adyacente CVD dE 9,1 / vision normal 22,9).
Dos de esos cuatro quedan por debajo de 3:1 de contraste contra el fondo, asi
que la regla de relieve obliga a **etiquetar cada serie directamente** ademas de
la leyenda: eso ya esta en `lineas()`.

Los graficos son PNG para las laminas del video, o sea superficie clara unica.
"""

from __future__ import annotations

import os
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

# --- tokens ---------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SUPERFICIE = "#ffffff"   # el fondo de la presentacion es blanco
TINTA = "#0b0b0b"
TINTA_2 = TINTA          # el texto va todo en negro; el gris se reserva para trazos
TINTA_MUDA = TINTA       # lineas de referencia y conectores, en negro
GRILLA = "#d6d5cf"       # la grilla NO va en negro: competiria con los datos

# El software de la presentacion quiere laminas un poco mas altas que anchas.
# Ojo: savefig usa bbox='tight', asi que la proporcion final depende de cuanto
# margen agreguen las etiquetas. Por eso `pie()` envuelve la nota al ancho
# de la figura: si no, el renglon del pie decide la proporcion.
VERTICAL = (7.4, 8.4)
EJE = TINTA              # ejes y linea de base, en negro
NEUTRO = "#f0efec"
ROJO = "#d03b3b"

#: Divergente azul <-> rojo con punto medio gris, para lifts centrados en 1.
CMAP_LIFT = LinearSegmentedColormap.from_list(
    "lift", ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRO, "#ec835a", ROJO, "#8f1f1f"]
)


def aplicar():
    """rcParams del proyecto. Llamar una vez al principio del notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EJE,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.titlecolor": TINTA,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TINTA_MUDA,
        "ytick.color": TINTA_MUDA,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 13,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def num(v, dec=0):
    """Numero a la castellana: punto para los miles, coma para los decimales."""
    s = f"{v:,.{dec}f}"
    return s.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


def _sin_marco(ax, eje_x=True):
    ax.spines["left"].set_visible(not eje_x)
    ax.spines["bottom"].set_visible(eje_x)
    ax.tick_params(length=0)


def barras_h(ax, etiquetas, valores, color=SERIES[0], dec=0, grosor=0.46):
    """Barras horizontales con el extremo del dato redondeado y la base a escuadra.

    `color` puede ser uno solo para todas, o una secuencia con uno por barra
    cuando las barras son categorias que ya tienen color propio en el resto de
    las figuras.

    El redondeo se consigue dibujando cada barra como una linea de extremo
    redondeado: el casquete de la izquierda cae fuera del `xlim`, que arranca en
    cero, asi que queda recortado a escuadra contra la linea de base. Es la forma
    de tener el radio en puntos (constante) y no en unidades de datos, que con
    ejes de escalas distintas deforma la barra.
    """
    valores = list(valores)
    colores = [color] * len(valores) if isinstance(color, str) else list(color)
    y = np.arange(len(etiquetas))[::-1]
    tope = max(valores) if valores else 1
    ax.set_yticks(y, etiquetas)
    ax.set_ylim(-0.7, len(etiquetas) - 0.3)
    ax.set_xlim(0, tope * 1.22)
    ax.set_xticks([])
    _sin_marco(ax, eje_x=False)
    ax.spines["left"].set_color(EJE)

    ax.figure.canvas.draw()                      # hace falta para medir el eje
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((0, grosor))
    puntos = abs(p1[1] - p0[1]) * 72 / ax.figure.dpi

    for yi, v, c in zip(y, valores, colores):
        ax.plot([0, max(v, tope * 0.002)], [yi, yi], color=c, linewidth=puntos,
                solid_capstyle="round", clip_on=True, zorder=2)
        ax.annotate(num(v, dec), (v, yi), xytext=(puntos / 2 + 7, 0),
                    textcoords="offset points", va="center", ha="left",
                    color=TINTA_2, fontsize=13.7)
    return ax


def envolver(etiquetas, ancho=11):
    """Parte las etiquetas en dos lineas para que entren debajo de una barra vertical."""
    return [textwrap.fill(e, ancho) for e in etiquetas]


def barras_v(ax, etiquetas, valores, color=SERIES[0], dec=0, grosor=0.46, ancho=11):
    """Barras verticales con el tope redondeado y la base a escuadra.

    Mismo truco que `barras_h` pero sobre el otro eje: cada barra es una linea de
    extremo redondeado cuyo casquete de abajo cae fuera del `ylim`, que arranca
    en cero, asi que queda recortado contra la linea de base.
    """
    valores = list(valores)
    colores = [color] * len(valores) if isinstance(color, str) else list(color)
    x = np.arange(len(etiquetas))
    tope = max(valores) if valores else 1
    ax.set_xticks(x, envolver(etiquetas, ancho) if ancho else etiquetas)
    ax.set_xlim(-0.65, len(etiquetas) - 0.35)
    ax.set_ylim(0, tope * 1.24)      # aire para que el numero no toque el titulo
    ax.set_yticks([])
    _sin_marco(ax, eje_x=True)
    ax.spines["bottom"].set_color(EJE)

    ax.figure.canvas.draw()                      # hace falta para medir el eje
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((grosor, 0))
    puntos = abs(p1[0] - p0[0]) * 72 / ax.figure.dpi

    for xi, v, c in zip(x, valores, colores):
        ax.plot([xi, xi], [0, max(v, tope * 0.002)], color=c, linewidth=puntos,
                solid_capstyle="round", clip_on=True, zorder=2)
        ax.annotate(num(v, dec), (xi, v), xytext=(0, puntos / 2 + 6),
                    textcoords="offset points", ha="center", va="bottom",
                    color=TINTA_2, fontsize=13.7, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SUPERFICIE, ec="none"))
    return ax


def lineas(ax, x, series, colores=None, fmt="{:.0f}", dx=0.10):
    """Lineas de 2px con marcador y etiqueta directa a la derecha.

    `series` es un dict nombre -> secuencia de valores. La etiqueta directa no es
    decoracion: dos de los cuatro colores no llegan a 3:1 contra el fondo, y la
    regla de relieve de la paleta pide que la identidad no dependa solo del color.

    Por eso la etiqueta directa es la unica: una leyenda ademas repetiria los
    mismos cuatro nombres y obligaria a leer dos veces lo mismo.
    """
    colores = colores or SERIES
    for (nombre, ys), c in zip(series.items(), colores):
        ax.plot(x, ys, color=c, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, label=nombre,
                solid_capstyle="round", zorder=3)

    # Las etiquetas directas se pisan cuando dos series terminan cerca. Se separan
    # verticalmente lo justo, en PUNTOS: la distancia minima legible no depende de
    # la escala del eje.
    finales = sorted(((ys[-1], nombre, c) for (nombre, ys), c in zip(series.items(), colores)),
                     key=lambda r: r[0])
    ax.figure.canvas.draw()
    y0, y1 = ax.get_ylim()
    alto_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    minimo = 15.5 * (y1 - y0) / alto_pt         # ~15 puntos entre lineas de texto
    colocadas = []
    for valor, nombre, c in finales:
        y = valor if not colocadas else max(valor, colocadas[-1] + minimo)
        colocadas.append(y)
        ax.annotate(f" {nombre}  {fmt.format(valor)}", (x[-1], y),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", color=c, fontsize=13, fontweight="bold")
    # Grilla en negro tenue, igual que el dumbbell: sobre fondo blanco el gris no
    # se ve, y las lineas de serie ya traen su propio color.
    ax.grid(axis="y", color=TINTA, linewidth=0.7, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    _sin_marco(ax)
    return ax


def referencia(ax, y, texto, lado="izq"):
    """Linea de referencia horizontal, en tinta muda y rotulada.

    `lado` ancla el rotulo al principio o al final del eje: con barras verticales
    el margen izquierdo suele estar ocupado y el derecho libre.
    """
    ax.axhline(y, color=TINTA_MUDA, linewidth=1.2, linestyle=(0, (5, 4)), zorder=1)
    x0, x1 = ax.get_xlim()
    x, dx, ha = (x0, 2, "left") if lado == "izq" else (x1, -2, "right")
    ax.annotate(texto, (x, y), xytext=(dx, 6), textcoords="offset points",
                color=TINTA, fontsize=12.3, va="bottom", ha=ha)


def pie(fig, texto):
    """Nota al pie: de donde salen los numeros.

    Se envuelve al ancho de la figura. Si no, `bbox="tight"` ensancha la imagen
    entera para que entre el renglon, y la proporcion vertical se pierde.
    """
    columnas = max(40, int(fig.get_size_inches()[0] * 14))
    fig.text(0.0, -0.035, textwrap.fill(texto, columnas), color=TINTA,
             fontsize=10, ha="left", va="top", linespacing=1.5)


def guardar(fig, nombre, carpeta="salidas/video_miranda/figuras"):
    """Guarda el png sobre el fondo del proyecto y una copia sin fondo en transparente/.

    La version transparente es para pegar en laminas con fondo propio. El halo
    blanco de los marcadores sigue siendo SUPERFICIE a proposito: sobre fondo
    claro separa las lineas, sobre fondo oscuro hay que rehacerlo.
    """
    os.makedirs(f"{carpeta}/transparente", exist_ok=True)
    fig.savefig(f"{carpeta}/{nombre}.png")
    fig.savefig(f"{carpeta}/transparente/{nombre}.png", transparent=True)
    return f"{carpeta}/{nombre}.png"
