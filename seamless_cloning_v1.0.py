# Importaciones necesarias
import cv2
import numpy as np 
 
# Leer imágenes
src = cv2.imread("person/0000063_06000_d_0000007_10.jpg")
dst = cv2.imread("output/0000063_06000_d_0000007.jpg")
 
# Crea una máscara aproximada alrededor del objeto.
src_mask = np.zeros(src.shape, src.dtype)
#poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
poly = np.array([
	[0, 0],
    [49, 0],
    [78, 49],
    [0, 78]
], np.int32)

cv2.fillPoly(src_mask, [poly], (255, 255, 255))
 
# Aquí es donde se colocará el CENTRO.
center = (660, 253)
 
# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
 
# Guardar resultados
cv2.imwrite("output/0000063_06000_d_0000007_NORMAL_CLONE.jpg", output)