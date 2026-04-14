from tqdm import tqdm
from glob import glob
from os.path import join
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import cv2
import config
import numpy as np
import os

def _poolCreation(args):
    root_data = args[0]
    anno_name = args[1]
    root_output = args[2]
    anno_name = anno_name.replace('\\', '/')
    filename = anno_name.split('/')[-1]
    
    # Cargar imagen original
    img_path = str(root_data/'images'/ filename.replace("txt", 'jpg'))
    img = cv2.imread(img_path)
    if img is None:
        return 0

    height_img, width_img = img.shape[:2]
    cont = 0

    with open(anno_name, 'r') as file: 
        for row in [x.split(' ') for x in file.read().strip().splitlines()]:
            if int(row[0]) == 0: # Clase Persona
                x = int(float(row[1]) * width_img) - int(float(row[3]) * width_img) // 2
                y = int(float(row[2]) * height_img) - int(float(row[4]) * height_img) // 2
                w = int(float(row[3]) * width_img)
                h = int(float(row[4]) * height_img)

                if x > 0 and y > 0 and w > 0 and h > 0:
                    crop_img = img[y:y+h, x:x+w]
                    # Guardar el recorte
                    cv2.imwrite(str(root_output/filename.replace(".txt", f'_{cont}.jpg')), crop_img) 
                    cont += 1
    return 0

def poolCreation(root_data, root_output, num_process=10):
    root_output.mkdir(parents=True, exist_ok=True) 
    
    # 1. Extraer los recortes (Lógica original)
    for rd in root_data:
        annos = glob(join((rd / 'labels'), '*.txt'))
        num_annos = len(annos)
        with Pool(processes=num_process) as pool:
            for n in tqdm(pool.imap_unordered(_poolCreation, zip([rd] * num_annos, annos, [root_output]*num_annos)),
                        desc='Generando recortes', total=num_annos, ncols=100):
                pass
    
    # 2. Cargar metadatos de VGGT para la vinculación
    # Asegúrate de tener ROOT_VGGT_METADATA definido en tu config.py
    vggt_csv_path = getattr(config, 'ROOT_VGGT_METADATA', 'camera_data_vx.csv')
    print(f"Cargando metadatos geométricos desde {vggt_csv_path}...")
    df_vggt = pd.read_csv(vggt_csv_path)

    # 3. Generar el CSV enriquecido
    list_images = []
    images_crops = glob(join(root_output, '*.jpg'))
    print("Vinculando recortes con datos de VGGT y guardando CSV...")
    
    for i in tqdm(images_crops, desc="Procesando metadatos"):
        img = cv2.imread(i)
        if img is None: continue
        h_crop, w_crop = img.shape[:2]
        
        # Obtener el nombre de la imagen original para buscar en VGGT
        # Ejemplo: 'path/imagen_original_0.jpg' -> 'imagen_original.jpg'
        crop_filename = os.path.basename(i)
        parts = crop_filename.split('_')
        original_img_base = "_".join(parts[:-1]) + ".jpg" # Ajustar extensión si es .png
        
        # Buscar la fila correspondiente en los datos de VGGT
        vggt_row = df_vggt[df_vggt['image_name'] == original_img_base]
        
        if not vggt_row.empty:
            row_data = vggt_row.iloc[0].to_dict()
            # Combinar datos del recorte con datos geométricos
            record = {
                'name': i,
                'height_patch': h_crop,
                'width_patch': w_crop,
                'original_image': original_img_base,
                **row_data # Incluye pos_z, height (relativa), depth_map_path, R_world_flat, etc.
            }
            list_images.append(record)
        else:
            # Si no hay coincidencia, guardar solo datos básicos
            list_images.append({'name': i, 'height_patch': h_crop, 'width_patch': w_crop})

    df_final = pd.DataFrame(list_images)
    df_final.to_csv(join(root_output, 'people_pool.csv'), index=False)
    print(f"Pool de personas creado con éxito en: {join(root_output, 'people_pool.csv')}")

if __name__ == '__main__':
    root_data = [Path(config.ROOT_DATA1)] 
    root_output = Path(config.ROOT_POOL_PERSON)
    poolCreation(root_data, root_output)