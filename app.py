"""Aplicacion web (Streamlit) del proyecto final: deteccion de resenas atipicas."""
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from resenas_modelo import cargar_modelo, probabilidad_calibrada, tokens_resaltables

st.set_page_config(page_title="Detector de reseñas atípicas", page_icon="🔎", layout="wide")

CARPETA = Path(__file__).parent / "modelo"
OPCION_LIBRE = "(escribir mi propia reseña)"
EJEMPLOS = {
    "Reseña genuina (inglés)": "We stayed here for five nights in October. The apartment is quiet, the bed was comfortable "
                               "and the host left us a map with restaurant tips. The only issue was the hot water in the "
                               "morning, but it was fixed the same day.",
    "Reseña genuina (español)": "Estuvimos cuatro noches en el apartamento. Está muy bien ubicado, a cinco minutos andando de "
                                "la plaza, y la anfitriona fue muy amable con el check-in. La cama es cómoda, aunque la calle "
                                "es un poco ruidosa por la noche.",
    "Elogio genérico muy corto": "Wonderful experience",
    "Texto de spam": "Click here to claim your free prize and earn money fast, limited offer for everyone!",
}


@st.cache_resource
def cargar():
    return cargar_modelo(CARPETA)


try:
    modelo, calib, mapa = cargar()
except FileNotFoundError:
    st.error("Faltan los archivos de la carpeta `modelo/`. Ejecuta `python entrenar_y_exportar.py` en la máquina del equipo "
             "y sube la carpeta `modelo/` al repositorio de GitHub.")
    st.stop()


def etiqueta(raiz):
    w = mapa.get(raiz, raiz)
    return raiz if w == raiz else f"{raiz} ({w})"


def resaltar(texto, contrib_por_raiz):
    """Texto original con las palabras coloreadas segun su contribucion al puntaje."""
    maximo = max([abs(v) for v in contrib_por_raiz.values()] + [1e-9])
    partes, pos = [], 0
    for ini, fin, raiz in tokens_resaltables(texto):
        partes.append(html.escape(texto[pos:ini]))
        palabra = html.escape(texto[ini:fin])
        c = contrib_por_raiz.get(raiz)
        if c is None:
            partes.append(palabra)
        else:
            alfa = 0.15 + 0.6 * min(abs(c) / maximo, 1.0)
            color = f"rgba(214,39,40,{alfa:.2f})" if c > 0 else f"rgba(31,119,180,{alfa:.2f})"
            partes.append(f"<span title='contribución {c:+.2f}' style='background:{color};padding:1px 3px;"
                          f"border-radius:3px'>{palabra}</span>")
        pos = fin
    partes.append(html.escape(texto[pos:]))
    return "<div style='white-space:pre-wrap;line-height:1.9;font-size:1.05rem'>" + "".join(partes) + "</div>"


def cargar_ejemplo():
    sel = st.session_state["ejemplo"]
    if sel != OPCION_LIBRE:
        st.session_state["texto"] = EJEMPLOS[sel]


st.title("🔎 Detector de reseñas atípicas")
st.caption("Proyecto final de Machine Learning · Modelo desplegado: Nivel 1 (TF-IDF + regresión logística)")
st.warning(
    "**Demostración académica.** El modelo se entrenó con patrones de fabricación *sintéticos* (reseñas copiadas, "
    "con ruido, plantillas de elogio y de spam) y su capacidad de detección es **muy limitada**: en el conjunto de "
    "prueba su precisión promedio (0.046) no supera a un ranking aleatorio (0.051). Un resultado de esta herramienta "
    "**no demuestra que una reseña sea falsa**; solo la prioriza para una revisión humana."
)

st.session_state.setdefault("texto", "")
st.session_state.setdefault("ejemplo", OPCION_LIBRE)

izq, der = st.columns([3, 1])
with izq:
    st.selectbox("Cargar un ejemplo", [OPCION_LIBRE] + list(EJEMPLOS), key="ejemplo", on_change=cargar_ejemplo)
    texto = st.text_area("Texto de la reseña", key="texto", height=150,
                         placeholder="Pega aquí una reseña (en inglés, español u otro idioma)…")
with der:
    anio = st.number_input("Año de la reseña", min_value=2010, max_value=2030, value=2024, step=1,
                           help="El modelo usa el año como una de sus variables; su peso es pequeño.")
    analizar = st.button("Analizar reseña", type="primary", use_container_width=True)

if analizar:
    if len(texto.strip()) < 3:
        st.error("Escribe una reseña de al menos unos caracteres.")
        st.stop()
    res = modelo.puntuar(texto, anio)
    cal = probabilidad_calibrada(res["score"], calib)

    st.subheader("Resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Decisión", "Marcar para revisión" if cal["marcar"] else "No marcar",
              help="Se marca el 5% de reseñas con mayor puntaje (el mismo punto de operación usado en la evaluación).")
    c2.metric("Percentil del puntaje", f"{cal['percentil']}",
              help="Posición del puntaje entre las reseñas del conjunto de validación (100 = el más sospechoso).")
    c3.metric("Probabilidad estimada de patrón de fabricación",
              f"{100 * cal['p']:.1f}%", f"IC 95%: {100 * cal['ic_inf']:.1f}% – {100 * cal['ic_sup']:.1f}%", delta_color="off")
    st.caption(
        f"La probabilidad es la **frecuencia observada** de positivos sintéticos entre {cal['n_tramo']:,} reseñas de validación "
        f"con un puntaje en el mismo tramo (prevalencia general: {100 * calib['prevalencia']:.0f}%). El intervalo se obtuvo por "
        "bootstrap por grupos. Como el modelo discrimina poco, estas probabilidades quedan cerca de la prevalencia."
    )
    if res["n_terminos_reconocidos"] == 0:
        st.info("Ninguna palabra de la reseña está en el vocabulario del modelo; el puntaje depende solo de las variables "
                "estructurales (longitud, mayúsculas, signos de exclamación y año).")

    st.subheader("¿Por qué este resultado?")
    st.markdown(
        "El modelo es lineal, así que su puntaje se descompone **exactamente** en: *intercepto + aporte de cada palabra + "
        "aporte de las variables estructurales*. Rojo = empuja hacia «patrón de fabricación»; azul = hacia «reseña original»."
    )
    contrib_raiz = {t: c for t, _, c in res["terminos"]}
    st.markdown(resaltar(texto, contrib_raiz), unsafe_allow_html=True)

    items = [(etiqueta(t), c) for t, _, c in res["terminos"]] + [(n, c) for n, _, c in res["num"]]
    items = [(n, c) for n, c in items if abs(c) > 1e-9]
    pos_ = sorted([x for x in items if x[1] > 0], key=lambda x: -x[1])[:8]
    neg_ = sorted([x for x in items if x[1] < 0], key=lambda x: x[1])[:8]
    top = (pos_ + neg_)
    if top:
        top = sorted(top, key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(7, 0.38 * len(top) + 1.0))
        ax.barh([n for n, _ in top], [c for _, c in top], color=["#d62728" if c > 0 else "#1f77b4" for _, c in top])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("aporte al puntaje (unidades de logit)")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    suma_t = sum(c for _, _, c in res["terminos"])
    suma_n = sum(c for _, _, c in res["num"])
    st.markdown(
        f"**Puntaje = intercepto ({res['intercepto']:+.2f}) + palabras ({suma_t:+.2f}) + variables estructurales "
        f"({suma_n:+.2f}) = {res['score']:+.2f}**  ·  probabilidad cruda del modelo (sin calibrar): {100 * res['prob_cruda']:.1f}%"
    )
    with st.expander("Detalle de las variables estructurales"):
        st.table({"variable": [n for n, _, _ in res["num"]],
                  "valor estandarizado": [f"{z:+.2f}" for _, z, _ in res["num"]],
                  "aporte al puntaje": [f"{c:+.2f}" for _, _, c in res["num"]]})

with st.expander("Acerca del modelo y sus límites"):
    st.markdown(
        """
**Modelo:** regresión logística sobre TF-IDF de unigramas (texto limpiado, con stopwords en español e inglés y stemming por idioma)
más 5 variables estructurales (longitud, signos de exclamación, mayúsculas, número de palabras y año). Hiperparámetros: C = 10,
`min_df` = 20, 50,000 términos como máximo. Se entrenó con todo el conjunto de entrenamiento.

**Por qué este modelo y no la red neuronal:** en el test reservado ambos tuvieron el mismo desempeño (AP 0.046 y 0.045), pero este
ocupa 0.4 MB frente a 15.7 MB, entrena más rápido, se despliega sin TensorFlow y su explicación es exacta y directa.

| Medida | Valor |
|---|---|
| AP en validación cruzada (5 pliegues) | 0.055 ± 0.012 |
| AP en el test reservado | 0.046 |
| AP de un ranking aleatorio (test) | 0.051 |
| AP de la regla «menor longitud» (test) | 0.075 |
| Precisión al marcar el 5% (test) | 0.037 (prevalencia 0.050) |

**Sesgo por idioma (tasa de falsos positivos sobre reseñas originales, test):** inglés 0.059, español 0.044, otros idiomas 0.015,
reseñas muy cortas 0.022. Las reseñas en inglés se marcan más que el promedio, probablemente porque las plantillas sintéticas
de entrenamiento están sobre todo en inglés (hipótesis no verificada).

**Límites:** los positivos de entrenamiento son sintéticos; el modelo no detecta copias de reseñas reales (un texto copiado es idéntico
a uno genuino); y el resultado no es una prueba de fraude. **Privacidad:** el texto ingresado solo se procesa en esta sesión y no se guarda.
        """
    )
