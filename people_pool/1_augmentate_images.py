# Aumento de personas en las imágenes.
# Se usan las imágenes generadas con el paso 0 y el paso 2
# Especificar: 
# - path de las imágenes de background (bgs)
# - path de la ubicación del pool de personas
# - la altura media de las personas
# - umbral de la altura para escoger a las personas
# USO
# python 1_augmentate_images.py



from tqdm import tqdm
from glob import glob
from os.path import join
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
#from skimage.exposure import match_histograms
import cv2
import random
import imutils
import numpy as np
import config


def convert_box_visdrone(size, box):
    # Convert VisDrone box to YOLO xywh box
    dw = 1. / size[1]
    dh = 1. / size[0]
    return (box[0] + box[2] / 2 - 1) * dw, (box[1] + box[3] / 2 - 1) * dh, box[2] * dw, box[3] * dh

def _augmentCrops(args):
    root_img = args[0]
    root_output = args[1]
    df_pool_people = args[2]

    root_img = root_img.replace('\\', '/')
    filename = root_img.split('/')[-1]

    img_bgs = cv2.imread(root_img)
    height = img_bgs.shape[0]
    width = img_bgs.shape[1]
    labels_file = root_img.replace('.jpg', '.txt').replace('images', 'labels')


    positions = []
    lines = []
    height_mean = 0
    num_rep_bg_crop = 1
    list_means_rep = []

    with open(labels_file, 'r') as file: 
        for row in [x.split(' ') for x in file.read().strip().splitlines()]:
            x = int(float(row[1]) * width)-int(float(row[3]) * width)//2
            y = int(float(row[2]) * height)-int(float(row[4]) * height)//2
            w = int(float(row[3]) * width)
            h = int(float(row[4]) * height)
            positions.append((x, y, w, h))
            
            box = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            lines.append(f"{0} {' '.join(f'{x:.6f}' for x in box)}\n") 
            height_mean += int(float(row[4]) * height)

    if height_mean == 0:
        for l in range(num_rep_bg_crop):
            height_mean = random.randrange(config.HEIGHT_MIN, config.HEIGHT_MAX)
            list_means_rep.append(height_mean)
    else:
        height_mean = height_mean // len(positions)
        list_means_rep.append(height_mean)

    if len(positions) < config.NUM_PEOPLE_X_IMG:
        num_persons_to_augmentate = config.NUM_PEOPLE_X_IMG
    else:
        num_persons_to_augmentate = len(positions)

    count = 0
    positions = []
    for height_mean in list_means_rep:

        lines_tempo = lines[:]
        filename=filename[:-4]+'_'+str(count)+filename[-4:]
        count += 1
        n = 0

        if height_mean == 0:
            cv2.imwrite(str(root_output/'images'/filename), img_bgs)
            with open(str(root_output/'labels'/filename.replace('.jpg', '.txt')), 'w') as fl:
                fl.writelines(lines_tempo)  # write label.txt
            return 0
        df_pool_people = df_pool_people[(df_pool_people['height'] > height_mean)]

        img_tempo = np.copy(img_bgs)
        while n < num_persons_to_augmentate:

            if len(df_pool_people) != 0:
                index_r = random.randrange(0, len(df_pool_people))
                img_person = cv2.imread(str(df_pool_people.iloc[index_r, 1]))
                if height_mean >= config.HEIGHT_MIN:
                    img_person = imutils.resize(img_person, height=random.randrange(height_mean-config.HEIGHT_AUG_LOW, height_mean+config.HEIGHT_AUG_HIGH))
                else:
                    img_person = imutils.resize(img_person, height=random.randrange(config.HEIGHT_MIN-config.HEIGHT_AUG_LOW, config.HEIGHT_MIN+config.HEIGHT_AUG_HIGH))
                
            else:
                print("No hay imágenes de personas en el pool")
                break

            h_persona = img_person.shape[0]
            w_persona = img_person.shape[1]
            range_img = [width, height]
            
            
            placed = False
            num_iter = 100
            while not placed:

                # Generar posición aleatoria
                if 0+w_persona < range_img[0]-w_persona:
                    x = random.randrange(0+w_persona,range_img[0]-w_persona)
                else:
                    break

                if 0+h_persona < range_img[1]-h_persona:
                    y = random.randrange(0+h_persona,range_img[1]-h_persona)
                else:
                    break

                # Comprobar si no hay solapamiento con las personas anteriores

                is_valid = True
                for (px, py, pw, ph) in positions:
                    if not (x + w_persona <= px or x >= px + pw or y + h_persona <= py or y >= py + ph):
                        is_valid = False
                        break
                
                if is_valid:
                    positions.append((x, y, w_persona, h_persona))
                    img_tempo[y:y+h_persona, x:x+w_persona]= img_person[:]
                    box = convert_box_visdrone(img_tempo.shape, tuple(map(int, [ x,y, w_persona, h_persona]))) 
                    lines_tempo.append(f"{0} {' '.join(f'{x:.6f}' for x in box)}\n")    
                
                if num_iter == 100:
                    break

                num_iter += 1
            
            n += 1
        
        cv2.imwrite(str(root_output/'images'/filename), img_tempo)
        with open(str(root_output/'labels'/filename.replace('.jpg', '.txt')), 'w') as fl:
            fl.writelines(lines_tempo)  # write label.txt
    return 0


def augmentCrops(root_bgs, root_pool_people, root_output, num_process=8):
    (root_output/'images').mkdir(parents=True, exist_ok=True) 
    (root_output/'labels').mkdir(parents=True, exist_ok=True) 

    bgs = glob(join(str(root_bgs/'images'), '*.jpg'))
    num_images = len(bgs)

    df_pool_people = pd.read_csv(str(root_pool_people/'pool.csv'))

    with Pool(processes=num_process) as pool:
        for n in tqdm(pool.imap_unordered(_augmentCrops, zip(bgs, [root_output]*num_images, [df_pool_people]*num_images)),
                      desc='process augs', total=num_images, ncols=100):
            pass


if __name__ == "__main__":
    for d in config.PARTITIONS:
        augmentCrops(Path(config.ROOT_DATA_AUG) / d, Path(config.ROOT_POOL_PERSON) / d, Path(config.ROOT_OUTPUT_AUG) / d)  
