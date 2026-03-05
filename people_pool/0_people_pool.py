# Creación de un pool de imágenes de personas.
# Se leen todas las imágenes de la base de datos (train y val) 
# y se extraen a las personas sin importar su tamaño.
# Especificar: 
# - path de las imágenes 
# - path donde se almacenará el pool de personas
# - el tamaño de los cortes
# USO
# python 0_people_crops_pool.py



from tqdm import tqdm
from glob import glob
from os.path import join
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import cv2
import config
import numpy as np



def _poolCreation(args):
    root_data = args[0]
    anno_name = args[1]
    root_output = args[2]
    anno_name = anno_name.replace('\\', '/')
    filename = anno_name.split('/')[-1]
    img = cv2.imread(str(root_data/'images'/ filename.replace("txt", 'jpg')))
    height = img.shape[0]
    width = img.shape[1]
    cont = 0
    #print('nombre: ', filename)

    with open(anno_name, 'r') as file: 
        for row in [x.split(' ') for x in file.read().strip().splitlines()]:
            if int(row[0]) == 0:
                x = int(float(row[1]) * width)-int(float(row[3]) * width)//2
                y = int(float(row[2]) * height)-int(float(row[4]) * height)//2
                w = int(float(row[3]) * width)
                h = int(float(row[4]) * height)     
                #print("x={}, y={}, w={}, h={}".format(x, y, w, h))
                if x > 0 and y > 0 and w > 0 and h > 0:
                    crop_img = img[y:y+h, x:x+w]
                    cv2.imwrite(str(root_output/filename.replace(".txt", f'_{cont}.jpg')), crop_img) 
                    cont += 1
        file.close()
        
    return 0


def poolCreation(root_data, root_output, num_process=10):
    (root_output).mkdir(parents=True, exist_ok=True) 
    
    
    for rd in root_data:
        annos = glob(join((rd / 'labels'), '*.txt'))
        num_annos = len(annos)
        with Pool(processes=num_process) as pool:
            for n in tqdm(pool.imap_unordered(_poolCreation, zip([rd] * num_annos, annos, [root_output]*num_annos)),
                        desc='process pool', total=num_annos, ncols=100):
                pass
    
    
    list_images = []
    images = glob(join(root_output, '*.jpg'))
    print("Guardando archivo csv...")
    for i in images:
        img = cv2.imread(str(i))
        height = img.shape[0]
        width = img.shape[1]
        list_images.append([i, height, width])
    dict = {'name': [row[0] for row in list_images], 'height':[row[1] for row in list_images], 'width':[row[2] for row in list_images]}
    df = pd.DataFrame(dict)
    df.to_csv(str(root_output/'pool.csv'))


if __name__ == "__main__":
    
    for d in config.PARTITIONS:
        #Path(config.ROOT_DATA2) / d,  Path(config.ROOT_DATA3) / d,
        poolCreation([Path(config.ROOT_DATA1) / d], Path(config.ROOT_POOL_PERSON) / d)  