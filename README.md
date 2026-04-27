# DA-Seamless-Cloning

Pipeline de **Data Augmentation** para datasets de detección de personas, combinando estimación de cámara con IA (VGGT) y fusión de imágenes por Seamless Cloning (OpenCV).

## Descripción

El proyecto extrae parámetros de cámara de imágenes reales usando el modelo **VGGT** (Facebook Research), agrupa las vistas en clusters, construye un pool de recortes de personas y los inserta en nuevas imágenes de fondo con restricción de profundidad mediante `cv2.seamlessClone`.

## Pipeline

```
1_vggt/1_extract_information.py   →   1_vggt/1_view_cluster.py
                                              ↓
                                 2_people_pool/1_people_pool.py
                                              ↓
                            3_data_augmentation/3_seamless_aug_depth.py
```

| Script | Función |
|---|---|
| `1_vggt/1_extract_information.py` | Extrae parámetros intrínsecos/extrínsecos de cámara y genera depth maps con VGGT |
| `1_vggt/1_view_cluster.py` | Visualiza en 3D los grupos de cámaras (clustering KMeans) |
| `2_people_pool/1_people_pool.py` | Recorta personas del dataset (YOLO) y genera `pool.csv` con metadatos de cámara |
| `3_data_augmentation/3_seamless_aug_depth.py` | Aplica Seamless Cloning guiado por profundidad y genera anotaciones YOLO |

### Scripts auxiliares

| Script | Función |
|---|---|
| `tools/video_to_frames.py` | Extrae frames de un video y los guarda como imágenes |
| `tools/yolo_person_labeler.py` | Herramienta de etiquetado manual/automático de personas en formato YOLO |

## Requisitos

### Entorno

- **Python** 3.13
- **CUDA** 13.0 (recomendado, GPU NVIDIA)
- **PyTorch** compatible con CUDA 13.0 (instalado desde `download.pytorch.org/whl/cu130`)

### Dependencias Python

| Paquete | Uso |
|---|---|
| `torch` / `torchvision` / `torchaudio` | Inferencia en GPU con VGGT |
| `numpy >= 2.0.0` | Procesamiento numérico general |
| `Pillow` | Lectura/escritura de imágenes |
| `matplotlib` | Visualización de depth maps y clusters 3D |
| `pandas` | Lectura/escritura de CSVs de metadatos |
| `scikit-learn` | Clustering KMeans de poses de cámara |
| `opencv-python` | Seamless Cloning, recorte de personas, I/O de video |
| `tqdm` | Barras de progreso en procesamiento por lotes |
| `huggingface_hub` | Descarga de pesos del modelo VGGT |
| `einops` | Operaciones de tensor requeridas por VGGT |
| `safetensors` | Carga de pesos en formato seguro |

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/pedroamtech/DA-Seamless-Cloning.git
cd DA-Seamless-Cloning

# 2. Crear y activar entorno conda
conda create -n dataaug python=3.13
conda activate dataaug

# 3. Instalar dependencias
pip install -r requirements_da.txt
```

> **Nota:** El paquete `vggt` ya viene incluido en la carpeta `1_vggt/vggt/` del repositorio. No es necesario instalarlo por separado.

## Estructura del Proyecto

```
DA-Seamless-Cloning/
├── 1_vggt/                          # Paso 1: Estimación de cámara con VGGT
│   ├── 1_extract_information.py     #   Extrae parámetros de cámara y depth maps
│   ├── 1_view_cluster.py            #   Visualización 3D de clusters de cámara
│   └── vggt/                        #   Repositorio VGGT (Facebook Research)
│
├── 2_people_pool/                   # Paso 2: Pool de recortes de personas
│   ├── 1_people_pool.py             #   Genera recortes y pool.csv con metadatos
│   ├── 0_people_pool.py             #   Versión básica (sin integración de metadatos)
│   └── config.py                    #   Configuración de rutas y parámetros
│
├── 3_data_augmentation/             # Paso 3: Augmentación con Seamless Cloning
│   └── 3_seamless_aug_depth.py      #   Inserción de personas guiada por profundidad
│
├── tools/                           # Herramientas auxiliares
│   ├── video_to_frames.py           #   Extracción de frames de video
│   └── yolo_person_labeler.py       #   Etiquetado automático en formato YOLO
│
├── back/                            # Respaldo (no forma parte del pipeline)
├── requirements_da.txt              # Dependencias del proyecto
└── README.md
```

## Uso

### 1. Extraer información de cámara

Selecciona la carpeta de imágenes y la carpeta de salida mediante el diálogo de archivos:

```bash
python 1_vggt/1_extract_information.py
```

Genera:
- `camera_data_vx.csv` — parámetros de cámara (posición, rotación, focal, altura relativa)
- `depth_maps/` — mapas de profundidad por imagen

### 2. Visualizar clusters de cámara

```bash
python 1_vggt/1_view_cluster.py
```

Muestra una visualización 3D interactiva con las poses de cámara agrupadas por KMeans. El número de clusters se configura mediante un diálogo al inicio.

### 3. Crear pool de personas

Configura las rutas en `2_people_pool/config.py` y ejecuta:

```bash
python 2_people_pool/1_people_pool.py
```

Genera `pool.csv` con los recortes de personas y sus metadatos de cámara (posición, focal, altura relativa) integrados automáticamente desde `camera_data_vx.csv`.

### 4. Augmentación con Seamless Cloning

```bash
python 3_data_augmentation/3_seamless_aug_depth.py
```

Inserta personas del pool en imágenes de fondo respetando la profundidad relativa, y genera las anotaciones YOLO correspondientes.

### 5. Herramientas auxiliares

**Extraer frames de video:**
```bash
python tools/video_to_frames.py
```

**Etiquetar/revisar personas manualmente:**
```bash
python tools/yolo_person_labeler.py
```

Controles del etiquetador: `ESPACIO` detectar | `A` procesar todo | `N`/`P` navegar | `S` guardar | `D` borrar | `Q` salir.

## Tecnologías

- **[VGGT](https://github.com/facebookresearch/vggt)** — Estimación de cámara y profundidad (Facebook Research)
- **PyTorch** + CUDA — Inferencia en GPU
- **OpenCV** (`cv2.seamlessClone`) — Fusión de imágenes y procesamiento de video
- **scikit-learn** — Clustering KMeans de poses de cámara
- **pandas / numpy** — Procesamiento de datos tabulares y numéricos
- **Hugging Face Hub** — Descarga de pesos del modelo VGGT-1B

## Configuración (`2_people_pool/config.py`)

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