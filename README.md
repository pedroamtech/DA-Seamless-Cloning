# DA-Seamless-Cloning

Pipeline de **Data Augmentation** para datasets de detección de personas, combinando estimación de cámara con IA (VGGT) y fusión de imágenes por Seamless Cloning (OpenCV).

## Descripción

El proyecto extrae parámetros de cámara de imágenes reales usando el modelo **VGGT** (Facebook), agrupa las vistas en clusters, construye un pool de recortes de personas y los inserta en nuevas imágenes de fondo con restricción de profundidad mediante `cv2.seamlessClone`.

## Pipeline

```
1_extract_information.py   →   1_view_cluster.py
                                      ↓
                           2_people_pool.py
                                      ↓
                           3_seamless_aug_depth.py
```

| Script | Función |
|---|---|
| `1_extract_information.py` | Extrae parámetros intrínsecos/extrínsecos de cámara y genera depth maps con VGGT |
| `1_view_cluster.py` | Visualiza en 3D los grupos de cámaras (clustering KMeans) |
| `2_people_pool.py` | Recorta personas del dataset (YOLO) y genera `pool.csv` con metadatos |
| `3_seamless_aug_depth.py` | Aplica Seamless Cloning guiado por profundidad y genera anotaciones YOLO |

### Scripts auxiliares

| Script | Función |
|---|---|
| `video_to_frames.py` | Extrae frames de un video y los guarda como imágenes |
| `yolo_person_labeler.py` | Herramienta de etiquetado manual de personas en formato YOLO |

## Requisitos

### Entorno

Python 3.13 + CUDA 13.0 (recomendado GPU NVIDIA)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/pedroam/DA-Seamless-Cloning.git
cd DA-Seamless-Cloning

# 2. Crear y activar entorno conda
conda create -n dataaug python=3.13
conda activate dataaug

# 3. Instalar dependencias
pip install -r requirements_da.txt
```

> **Nota:** El paquete `vggt` ya viene incluido en la carpeta `vggt/` del repositorio. No es necesario instalarlo por separado.

## Estructura del Proyecto

```
DA-Seamless-Cloning/
├── 1_extract_information.py     # Paso 1: Extracción de parámetros de cámara
├── 1_view_cluster.py            # Paso 1b: Visualización de clusters de cámara
├── 2_people_pool.py             # Paso 2: Creación del pool de personas
├── 3_seamless_aug_depth.py      # Paso 3: Augmentación con seamless cloning
├── video_to_frames.py           # Aux: Extracción de frames de video
├── yolo_person_labeler.py       # Aux: Etiquetado manual en formato YOLO
├── requirements_da.txt          # Dependencias del proyecto
├── vggt/                        # Repositorio VGGT (Facebook)
└── people_pool/
    └── config.py                # Configuración de rutas y parámetros
```

## Uso

### 1. Extraer información de cámara

Selecciona la carpeta de imágenes y la carpeta de salida mediante el diálogo de archivos:

```bash
python 1_extract_information.py
```

Genera:
- `camera_data_vx.csv` — parámetros de cámara (posición, rotación, focal, altura relativa)
- `depth_maps/` — mapas de profundidad por imagen

### 2. Visualizar clusters de cámara

```bash
python 1_view_cluster.py
```

Muestra una visualización 3D interactiva con las poses de cámara agrupadas.

### 3. Crear pool de personas

Configura las rutas en `people_pool/config.py` y ejecuta:

```bash
python 2_people_pool.py
```

Genera `pool.csv` con los recortes de personas y sus metadatos de cámara.

### 4. Augmentación con Seamless Cloning

```bash
python 3_seamless_aug_depth.py
```

Inserta personas del pool en imágenes de fondo respetando la profundidad relativa, y genera las anotaciones YOLO correspondientes.

## Tecnologías

- **[VGGT](https://github.com/facebookresearch/vggt)** — Estimación de cámara y profundidad
- **PyTorch** + CUDA — Inferencia en GPU
- **OpenCV** (`cv2.seamlessClone`) — Fusión de imágenes
- **scikit-learn** — Clustering KMeans de poses de cámara
- **pandas / numpy** — Procesamiento de datos

## Configuración (`people_pool/config.py`)

| Variable | Descripción |
|---|---|
| `ROOT_DATA1` | Ruta raíz del dataset |
| `ROOT_POOL_PERSON` | Carpeta de salida del pool |
| `ROOT_OUTPUT_AUG` | Carpeta de salida de imágenes augmentadas |
| `ROOT_VGGT_METADATA` | Ruta al `camera_data_vx.csv` |
| `PARTITIONS` | Lista de particiones (e.g. `['train', 'val']`) |
| `NUM_PEOPLE_X_IMG` | Personas a insertar por imagen |

## Autor

**Pedro AM** · [@pedroamtech](https://github.com/pedroamtech)

---

⭐ Si este proyecto te ha sido útil, ¡dale una estrella!