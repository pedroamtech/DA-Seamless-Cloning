# DA-Seamless-Cloning

Pipeline de **Data Augmentation** para datasets de detección de personas, combinando estimación de cámara con IA (VGGT) y fusión de imágenes por Seamless Cloning (OpenCV).

## Descripción

El proyecto extrae parámetros de cámara de imágenes reales usando el modelo **VGGT** (Facebook Research), agrupa las vistas en clusters, construye un pool de recortes de personas y los inserta en nuevas imágenes de fondo con restricción de profundidad mediante `cv2.seamlessClone`.

## Pipeline

```
1_extract_information.py   →   2_view_cluster.py
                                      ↓
                               3_people_pool.py
                                      ↓
                         4_seamless_aug_depth_v6.py
```

| Script | Función |
|---|---|
| `1_extract_information.py` | Extrae parámetros de cámara y genera depth maps (Visual y RAW) con VGGT. Soporta CLI y HF_TOKEN. |
| `2_view_cluster.py` | Visualiza en 3D los grupos de cámaras (clustering KMeans). |
| `3_people_pool.py` | Recorta personas del dataset (YOLO) y genera `pool.csv` con metadatos de cámara integrados. |
| `4_seamless_aug_depth_v6.py` | **V6 Definitiva:** Aplica Seamless Cloning realista usando escalado métrico absoluto, enmascaramiento semántico dinámico para evitar fondos inválidos, y Edge Padding para conservar la textura de las personas. |

### Scripts auxiliares

| Script | Función |
|---|---|
| `tools/video_to_frames.py` | Extrae frames de un video y los guarda como imágenes. |
| `tools/yolo_person_labeler.py` | Herramienta de etiquetado manual/automático de personas en formato YOLO. |

## Requisitos

### Entorno

- **Python** 3.13
- **CUDA** 13.0 (recomendado, GPU NVIDIA)
- **PyTorch** compatible con CUDA 13.0.

### Dependencias Principales

`torch`, `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `matplotlib`, `tqdm`, `huggingface_hub`, `einops`.

Instalar mediante:
```bash
pip install -r requirements_da.txt
```

### Autenticación Hugging Face (Recomendado)
Para evitar límites de velocidad y avisos, sigue estos pasos:
1. **Crear cuenta:** Regístrate en [huggingface.co/join](https://huggingface.co/join).
2. **Generar Token:** Ve a [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) y crea un token de tipo **Read**.
3. **Configurar en Windows:** Ejecuta en PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'TU_TOKEN_AQUI', 'User')
```

## Estructura del Proyecto

```
DA-Seamless-Cloning/
├── 1_extract_information.py     # Paso 1: Extracción de cámara y profundidad
├── 2_view_cluster.py            # Paso 2: Visualización de clusters
├── 3_people_pool.py             # Paso 3: Creación del pool de personas
├── 4_seamless_aug_depth_v6.py   # Paso 4: Augmentación (V6 Definitiva)
├── config.py                    # Configuración global de rutas
├── vggt/                        # Código fuente del modelo VGGT
├── people_pool/                 # Scripts de apoyo para el pool
├── tools/                       # Herramientas de video y etiquetado
├── back/                        # Respaldos de versiones anteriores
└── README.md
```

## Uso

### 1. Extraer información de cámara
Puedes usar el diálogo interactivo o pasar las rutas por CLI:
```bash
python 1_extract_information.py --img_dir "ruta/fotos" --out_dir "ruta/salida" --batch 50
```

### 2. Crear pool de personas
Configura las rutas en `config.py` y ejecuta:
```bash
python 3_people_pool.py
```

### 3. Augmentación con Profundidad (V6 Definitiva)
Este script busca automáticamente los mapas de profundidad RAW (`depth_*.png`) generados en el paso 1 para realizar un pegado geométricamente coherente, con ajuste de escala métrica y filtrado de resolución.
```bash
python 4_seamless_aug_depth_v6.py
```

## Tecnologías

- **[VGGT](https://github.com/facebookresearch/vggt)** — Estimación de cámara y profundidad (Facebook Research).
- **OpenCV** (`cv2.seamlessClone`) — Fusión de imágenes realista.
- **Hugging Face Hub** — Gestión del modelo pre-entrenado VGGT-1B.

## Autor

**Pedro AM** · [@pedroamtech](https://github.com/pedroamtech)

---
⭐ Si este proyecto te ha sido útil, ¡dale una estrella!