# Detector de reseñas atípicas — aplicación web

Proyecto final de Machine Learning (Parte 8: despliegue). Aplicación web en **Streamlit** que recibe el texto de una
reseña y devuelve: la decisión (marcar o no para revisión), el percentil del puntaje, una **probabilidad estimada con su
intervalo de confianza del 95%** y una **explicación de la predicción** para el caso ingresado (palabras que empujan hacia
cada clase y descomposición exacta del puntaje).

## URL pública

** https://detector-resenas-atipicas.streamlit.app/ ** ← *pegar aquí la URL real después de desplegar*

## Modelo desplegado

Nivel 1 del proyecto: **TF-IDF + regresión logística** (Parte 3), entrenado con todo el conjunto de entrenamiento.
Se eligió sobre la red neuronal (Parte 5) porque, en el test reservado, ambos tienen el mismo desempeño
(AP 0.046 frente a 0.045; la diferencia incluye el cero), pero el modelo clásico ocupa 0.4 MB frente a 15.7 MB, entrena más rápido,
no requiere TensorFlow para desplegarse y su explicación es exacta (el puntaje se descompone en la suma de los aportes de cada palabra).
No se eligió por tener la mejor métrica: **ningún modelo supera de forma relevante a un ranking aleatorio**, por lo que la aplicación
advierte que es una demostración académica.

La inferencia de la app está escrita en Python puro (`resenas_modelo.py`) y lee los parámetros del modelo desde `modelo/*.json`.
Así **no depende de scikit-learn** y se evitan los problemas de versiones al desplegar.

## Estructura

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit (interfaz, resultado y explicación). |
| `resenas_modelo.py` | Limpieza de texto, cálculo del puntaje y descomposición (sin scikit-learn). |
| `modelo/modelo_nivel1.json` | Parámetros del modelo: vocabulario, IDF, coeficientes, estandarización e intercepto. |
| `modelo/calibracion.json` | Tabla que convierte el puntaje en probabilidad empírica con IC 95% (validación cruzada de la Parte 6). |
| `modelo/mapa_stems.json` | Raíz → palabra original más frecuente (solo para mostrar términos legibles). |
| `modelo/info.json` | Versiones, tamaño del vocabulario y huella del modelo exportado. |
| `entrenar_y_exportar.py` | Entrena el modelo y genera los JSON (se ejecuta una vez, en la máquina del equipo). |
| `requirements.txt` | Dependencias de la app desplegada. |
| `requirements-entrenamiento.txt` | Dependencias solo para reentrenar/exportar. |

## Reproducir en local

```bash
# 1) (opcional) reentrenar y exportar el modelo; requiere salidas/train.csv.gz y salidas/parte6_oof.csv
python -m venv .venv && source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements-entrenamiento.txt
python entrenar_y_exportar.py --train salidas/train.csv.gz --oof salidas/parte6_oof.csv

# 2) ejecutar la aplicación
pip install -r requirements.txt
streamlit run app.py
```

El script verifica que la inferencia exportada coincide con el pipeline de scikit-learn (diferencia máxima < 1e-6) y se detiene si no es así.

## Despliegue en Streamlit Community Cloud

1. Crear un repositorio **público** en GitHub y subir: `app.py`, `resenas_modelo.py`, `requirements.txt`, `README.md`, `.gitignore`
   y la carpeta `modelo/` con sus cuatro JSON (el repositorio queda por debajo de 2 MB; los datos de entrenamiento **no** se suben).
2. Entrar a <https://share.streamlit.io> e iniciar sesión con la cuenta de GitHub (autorizar a Streamlit si lo pide).
3. Pulsar **Create app** (arriba a la derecha) y elegir desplegar una app existente desde GitHub.
4. Completar **Repository**, **Branch** (`main`) y **Main file path** (`app.py`). Opcional: elegir un subdominio.
5. En **Advanced settings** dejar la versión de Python por defecto (la app no depende de la versión).
6. Pulsar **Deploy**. La primera instalación tarda unos minutos.
7. Copiar la URL pública (`https://….streamlit.app`), abrirla en una ventana privada para confirmar que funciona sin sesión,
   y pegarla en la sección «URL pública» de este README y en el informe.

Nota: las apps gratuitas se hibernan tras un período sin visitas; al abrirlas, un botón permite reactivarlas. Conviene abrir la URL
antes de la entrega para verificar que está activa.

## Cómo se calcula lo que muestra la app

* **Puntaje:** `intercepto + Σ (coeficiente × valor TF-IDF de cada palabra) + Σ (coeficiente × variable estructural estandarizada)`.
* **Decisión:** se marca la reseña si su puntaje está en el 5% más alto de las predicciones fuera de muestra de la validación cruzada.
* **Probabilidad e intervalo:** frecuencia observada de positivos en el tramo de puntaje (20 tramos), con IC 95% por bootstrap por grupos.
* **Explicación:** descomposición exacta del puntaje, con el texto coloreado y un gráfico de los aportes principales.

## Límites

* Los positivos de entrenamiento son **sintéticos**; la probabilidad se refiere a esos patrones, no a fraude real.
* En el test reservado el AP es 0.046 (azar: 0.051); el modelo no detecta copias de reseñas reales.
* Las reseñas originales en inglés se marcan más que las de otros idiomas (tasa de falsos positivos 0.059 frente a 0.044 en español
  y 0.015 en otros idiomas, en el test).
