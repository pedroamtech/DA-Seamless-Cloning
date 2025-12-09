# Importaciones necesarias
import cv2
import numpy as np 

# Leer imágenes
src = cv2.imread("person/9999962_00000_d_0000006_17.jpg")
dst = cv2.imread("images/9999962_00000_d_0000006.jpg")

# Crea una máscara aproximada alrededor del objeto
src_mask = np.zeros(src.shape[:2], dtype=np.uint8)  # Máscara en escala de grises
poly = np.array([
    [0, 0],
    [50, 0],
    [100, 50],
    [0, 100]
], np.int32)

cv2.fillPoly(src_mask, [poly], 255)  # Llenar con blanco (255)

# Posición donde se colocará (esquina superior izquierda)
x_offset = 1056 - src.shape[1]//2  # Centrar horizontalmente
y_offset = 728 - src.shape[0]//2   # Centrar verticalmente

# Asegurar que no se salga de los límites
x_offset = max(0, min(x_offset, dst.shape[1] - src.shape[1]))
y_offset = max(0, min(y_offset, dst.shape[0] - src.shape[0]))

# Crear una copia de la imagen destino
output = dst.copy()

# Aplicar la máscara y pegar la región
for y in range(src.shape[0]):
    for x in range(src.shape[1]):
        if src_mask[y, x] > 0:  # Si el pixel está en la máscara
            if (y + y_offset < dst.shape[0] and x + x_offset < dst.shape[1]):
                output[y + y_offset, x + x_offset] = src[y, x]

# Guardar resultado
cv2.imwrite("output/0000013_00465_d_0000067_SIMPLE_PASTE.jpg", output)