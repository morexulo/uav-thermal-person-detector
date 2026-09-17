# UAV Thermal Person Detector

Detección de personas en imágenes térmicas capturadas desde dron, pensada para escenarios de **búsqueda y rescate (SAR)** en entornos rurales (pistas de aterrizaje, caminos de tierra, campos abiertos).

Modelos basados en YOLOv8, entrenados sobre un dataset propio de imágenes térmicas (con algunos frames RGB) etiquetadas con una única clase: `person`.

## Modelos

Se entrenaron dos variantes con objetivos distintos:

| Modelo | Arquitectura | Params | imgsz | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| `best_uav_fast.pt` | YOLOv8s | 9.8M | 512 | 0.822 | 0.607 | 0.693 | 0.424 |
| `monet_pop_p2_1024` (best epoch) | YOLOv8m-**p2** | 25.1M | 960 | 0.889 | 0.713 | – | 0.502 |

- **`best_uav_fast`**: modelo ligero, pensado para inferencia rápida (~76 ms/frame en CPU a 512px). Buen punto de partida si el hardware del dron/embedded es limitado.
- **`monet_pop_p2_1024`** (ver [`notebooks/train_p2_1024.ipynb`](notebooks/train_p2_1024.ipynb)): variante con cabeza de detección adicional **P2** (stride 4), pensada específicamente para objetos muy pequeños en el frame — una persona vista desde altura suele ocupar muy pocos píxeles. Entrenado a mayor resolución (960px), consigue mejor recall y mAP50-95 a costa de más cómputo. *Los pesos de este modelo no están incluidos en este repo (entrenado y guardado en Google Drive); solo se incluye el notebook de entrenamiento y las métricas.*

Solo `best_uav_fast.pt` está incluido en [`models/`](models/) por tamaño y disponibilidad.

## Dataset

- Imágenes térmicas reales de dron (ruido de sensor tipo FLIR: bandas horizontales, viñeteado), combinadas con algunos frames RGB.
- Organizado por escenario de captura: pistas de aterrizaje (`runway`), caminos de tierra (`dirtroad`), etc.
- Clase única: `person`. Las cajas etiquetadas son mayoritariamente muy pequeñas (personas vistas desde altura), lo que hace del recall la métrica más difícil de mejorar.
- Ejemplos de batches de entrenamiento (mosaic augmentation) en [`samples/`](samples/):

  ![train_batch0](samples/train_batch0.jpg)

## Limitaciones

- El mAP50-95 (~0.42–0.50) indica que la localización exacta de la caja no es perfecta, aunque la detección en sí (mAP50) es razonable.
- El recall del modelo `fast` (0.61) implica que se le escapan personas con cierta frecuencia — **no debe usarse como sistema autónomo de decisión**, sino como herramienta de apoyo/cribado para un operador humano.
- Dataset con pocas cajas por imagen y muchos fondos sin persona (hard negatives), típico de escenarios SAR reales donde el positivo es raro.

## Uso rápido

```bash
pip install ultralytics

# Inferencia sobre una imagen o carpeta
yolo predict model=models/best_uav_fast.pt source=ruta/a/imagen_o_carpeta conf=0.25

# Inferencia en vivo desde webcam/cámara
yolo predict model=models/best_uav_fast.pt source=0 show=True
```

O desde Python:

```python
from ultralytics import YOLO

model = YOLO("models/best_uav_fast.pt")
results = model.predict("ruta/a/imagen.jpg", conf=0.25)
results[0].show()
```

## Entrenamiento

El notebook [`notebooks/train_p2_1024.ipynb`](notebooks/train_p2_1024.ipynb) contiene el pipeline completo de entrenamiento de la variante YOLOv8m-p2 en Google Colab (montaje de Drive, extracción del dataset, `yolo train` con `imgsz=960`, `cos_lr`, resume tras interrupciones de sesión, y análisis de métricas por época).

## Licencia

[MIT](LICENSE)
