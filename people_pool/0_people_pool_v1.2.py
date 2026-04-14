# Creación de un pool de imágenes de personas.
# Se leen todas las imágenes de la base de datos (train y val)
# y se extraen las personas sin importar su tamaño.
#
# Cambios v1.2:
#   - Los metadatos del CSV de cámara (camera_data_vx.csv) ubicado
#     en la ruta de imágenes del dataset (ROOT_VGGT_METADATA en config.py)
#     se integran automáticamente en el CSV final del pool.
#   - La unión se realiza por nombre de imagen original (image_name).
#   - Si una imagen fuente no tiene metadatos, se guardan igualmente
#     los datos básicos del recorte (name, height_patch, width_patch).


from tqdm import tqdm
from glob import glob
from os.path import join, basename
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import pandas as pd
import cv2
import config
import numpy as np


# ---------------------------------------------------------------------------
# Función worker: genera los recortes de personas de UNA imagen.
# Devuelve una lista de dicts con info básica de cada recorte.
# ---------------------------------------------------------------------------
def _poolCreation(args):
    root_data   = args[0]   # Path al directorio raíz de la partición
    anno_name   = args[1]   # Ruta del .txt de anotaciones YOLO
    root_output = args[2]   # Path al directorio de salida del pool

    anno_name = anno_name.replace('\\', '/')
    filename  = anno_name.split('/')[-1]          # e.g. '0000059_01886_d_0000114.txt'

    img_path = str(root_data / 'images' / filename.replace('.txt', '.jpg'))
    img = cv2.imread(img_path)
    if img is None:
        return []

    height_img, width_img = img.shape[:2]
    cont      = 0
    crops_info = []   # lista de dicts con info de cada recorte

    with open(anno_name, 'r') as file:
        for row in [x.split(' ') for x in file.read().strip().splitlines()]:
            if not row or len(row) < 5:
                continue
            if int(row[0]) == 0:   # Clase 0 = Persona
                x = int(float(row[1]) * width_img)  - int(float(row[3]) * width_img)  // 2
                y = int(float(row[2]) * height_img) - int(float(row[4]) * height_img) // 2
                w = int(float(row[3]) * width_img)
                h = int(float(row[4]) * height_img)

                if x > 0 and y > 0 and w > 0 and h > 0:
                    crop_img  = img[y:y+h, x:x+w]
                    crop_name = filename.replace('.txt', f'_{cont}.jpg')
                    crop_path = str(root_output / crop_name)
                    cv2.imwrite(crop_path, crop_img)

                    crops_info.append({
                        'name':           crop_path,
                        'height_patch':   h,
                        'width_patch':    w,
                        'original_image': filename.replace('.txt', '.jpg'),  # clave de unión con el CSV
                    })
                    cont += 1

    return crops_info


# ---------------------------------------------------------------------------
# Función principal: genera recortes y construye el CSV enriquecido.
# ---------------------------------------------------------------------------
def poolCreation(root_data_list, root_output, num_process=10):
    root_output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Cargar el CSV de metadatos de cámara
    # ------------------------------------------------------------------
    meta_csv_path = getattr(config, 'ROOT_VGGT_METADATA', None)
    if meta_csv_path is None:
        raise ValueError(
            "ROOT_VGGT_METADATA no está definido en config.py. "
            "Añade la ruta del archivo camera_data_vx.csv a config.py."
        )

    print(f"\n[INFO] Cargando metadatos de cámara desde:\n       {meta_csv_path}")
    df_meta = pd.read_csv(meta_csv_path)

    # Normalizar la columna de nombre de imagen (sin ruta, solo basename)
    df_meta['image_name'] = df_meta['image_name'].apply(basename)

    # Indexar por nombre de imagen para búsqueda rápida O(1)
    df_meta_indexed = df_meta.set_index('image_name')

    print(f"[INFO] CSV de metadatos cargado: {len(df_meta)} filas | "
          f"columnas: {df_meta.columns.tolist()}\n")

    # ------------------------------------------------------------------
    # 2. Generar los recortes con multiprocesamiento
    # ------------------------------------------------------------------
    all_crops = []   # acumula los dicts de cada recorte

    for rd in root_data_list:
        annos     = glob(join(str(rd / 'labels'), '*.txt'))
        num_annos = len(annos)
        print(f"[INFO] Procesando {num_annos} anotaciones en: {rd}")

        with Pool(processes=num_process) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _poolCreation,
                        zip([rd] * num_annos, annos, [root_output] * num_annos)
                    ),
                    desc='Generando recortes',
                    total=num_annos,
                    ncols=100,
                )
            )

        # Aplanar lista de listas
        for crop_list in results:
            all_crops.extend(crop_list)

    # ------------------------------------------------------------------
    # 3. Construir el CSV final enriquecido con los metadatos de cámara
    # ------------------------------------------------------------------
    print(f"\n[INFO] Total de recortes generados: {len(all_crops)}")
    print("[INFO] Vinculando recortes con metadatos de cámara...")

    records = []
    matched   = 0
    unmatched = 0

    for crop in tqdm(all_crops, desc='Integrando metadatos', ncols=100):
        orig_img = crop['original_image']   # e.g. '0000059_01886_d_0000114.jpg'
        record   = dict(crop)               # copia de los datos básicos

        if orig_img in df_meta_indexed.index:
            # Añadir todas las columnas del CSV de metadatos
            meta_row = df_meta_indexed.loc[orig_img].to_dict()
            record.update(meta_row)
            matched += 1
        else:
            unmatched += 1

        records.append(record)

    # ------------------------------------------------------------------
    # 4. Guardar CSV (con fallback si el archivo está abierto en Excel)
    # ------------------------------------------------------------------
    df_final = pd.DataFrame(records)
    output_csv = root_output / 'pool.csv'

    def _save_csv(path):
        df_final.to_csv(str(path), index=False)
        return path

    try:
        saved_path = _save_csv(output_csv)
    except PermissionError:
        # El archivo está abierto en otro programa → usar nombre con timestamp
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = root_output / f'pool_{ts}.csv'
        print(f"\n[AVISO] pool.csv está en uso (¿abierto en Excel?). "
              f"Guardando en nombre alternativo...")
        saved_path = _save_csv(output_csv)

    print(f"\n[INFO] CSV guardado en: {saved_path}")
    print(f"       Recortes con metadatos de cámara      : {matched}")
    print(f"       Recortes sin metadatos (solo básico)  : {unmatched}")
    print(f"       Total de filas en el CSV              : {len(df_final)}")
    print(f"       Columnas del CSV                      : {df_final.columns.tolist()}")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    for d in config.PARTITIONS:
        poolCreation(
            root_data_list=[Path(config.ROOT_DATA1) / d],
            root_output=Path(config.ROOT_POOL_PERSON) / d,
        )
