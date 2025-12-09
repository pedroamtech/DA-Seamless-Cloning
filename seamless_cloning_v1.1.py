import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os

def create_automatic_mask(src_img, method='threshold', threshold_value=240):
    """
    Crea una máscara automática usando diferentes métodos
    
    Args:
        src_img: Imagen fuente
        method: 'threshold', 'grabcut', 'contour'
        threshold_value: Valor de umbral para threshold
    """
    if method == 'threshold':
        # Método de umbralización (más simple y rápido)
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # Operaciones morfológicas para limpiar la máscara
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Suavizar bordes
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        
        # Convertir a 3 canales
        mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
    elif method == 'grabcut':
        # Método GrabCut (más preciso pero más lento)
        mask = np.zeros(src_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        
        # Definir rectángulo inicial (ajustar según necesidad)
        height, width = src_img.shape[:2]
        rect = (10, 10, width-20, height-20)
        
        cv2.grabCut(src_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Crear máscara final
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask = cv2.cvtColor(mask2 * 255, cv2.COLOR_GRAY2BGR)
        
    elif method == 'contour':
        # Método de contornos
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear máscara basada en el contorno más grande
        mask = np.zeros(src_img.shape, dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], (255, 255, 255))
    
    return mask

def poisson_blend_channel(source, target, mask, offset_x, offset_y):
    """
    Implementación básica de Poisson blending para un canal
    Basado en el paper "Poisson Image Editing"
    """
    h, w = source.shape
    target_h, target_w = target.shape
    
    # Crear matriz para el sistema lineal
    num_pixels = h * w
    A = diags([1, -4, 1, 1, 1], [-w, 0, w, -1, 1], shape=(num_pixels, num_pixels), format='csr')
    
    # Vector de términos independientes
    b = np.zeros(num_pixels)
    
    # Llenar el sistema basado en la ecuación de Poisson
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            target_i = i + offset_y
            target_j = j + offset_x
            
            # Verificar límites
            if (target_i < 0 or target_i >= target_h or 
                target_j < 0 or target_j >= target_w):
                continue
            
            # Si está en el borde de la máscara, usar valor del target
            if mask[i, j] == 0:
                A[idx, idx] = 1
                b[idx] = target[target_i, target_j]
            else:
                # Calcular gradiente de la fuente
                grad_sum = 0
                neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                
                for ni, nj in neighbors:
                    if 0 <= ni < h and 0 <= nj < w:
                        grad_sum += source[i,j] - source[ni,nj]
                
                b[idx] = grad_sum
    
    # Resolver sistema lineal
    try:
        result = spsolve(A, b)
        return result.reshape((h, w))
    except:
        # Fallback a método simple si falla
        return source

def enhanced_seamless_clone(src, dst, mask, center, blend_type='mixed', use_poisson=False):
    """
    Versión mejorada de seamless cloning con múltiples opciones
    
    Args:
        src: Imagen fuente
        dst: Imagen destino  
        mask: Máscara
        center: Centro donde colocar
        blend_type: 'normal', 'mixed', 'feature_exchange', 'color_change'
        use_poisson: Si usar implementación propia de Poisson o OpenCV
    """
    
    if use_poisson:
        # Usar implementación propia de Poisson blending
        print("Usando implementación propia de Poisson blending...")
        
        # Calcular offset
        h, w = src.shape[:2]
        offset_x = center[0] - w // 2
        offset_y = center[1] - h // 2
        
        # Convertir máscara a escala de grises
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        # Aplicar blending canal por canal
        result = dst.copy()
        for c in range(3):
            blended_channel = poisson_blend_channel(
                src[:,:,c], dst[:,:,c], mask_gray, offset_x, offset_y
            )
            
            # Copiar resultado al área correspondiente
            y1, y2 = max(0, offset_y), min(dst.shape[0], offset_y + h)
            x1, x2 = max(0, offset_x), min(dst.shape[1], offset_x + w)
            
            if y2 > y1 and x2 > x1:
                src_y1 = max(0, -offset_y)
                src_x1 = max(0, -offset_x)
                src_y2 = src_y1 + (y2 - y1)
                src_x2 = src_x1 + (x2 - x1)
                
                result[y1:y2, x1:x2, c] = blended_channel[src_y1:src_y2, src_x1:src_x2]
        
        return result
    
    else:
        # Usar OpenCV seamlessClone
        clone_flags = {
            'normal': cv2.NORMAL_CLONE,
            'mixed': cv2.MIXED_CLONE,
            'feature_exchange': cv2.MIXED_CLONE,  # OpenCV no tiene FEATURE_EXCHANGE
            'color_change': cv2.MIXED_CLONE
        }
        
        flag = clone_flags.get(blend_type, cv2.MIXED_CLONE)
        return cv2.seamlessClone(src, dst, mask, center, flag)

def multi_scale_blending(src, dst, mask, center, scales=[0.5, 1.0, 1.5]):
    """
    Aplica blending a múltiples escalas y combina resultados
    """
    results = []
    
    for scale in scales:
        if scale != 1.0:
            # Redimensionar
            new_size = (int(src.shape[1] * scale), int(src.shape[0] * scale))
            src_scaled = cv2.resize(src, new_size, interpolation=cv2.INTER_CUBIC)
            mask_scaled = cv2.resize(mask, new_size, interpolation=cv2.INTER_CUBIC)
        else:
            src_scaled = src.copy()
            mask_scaled = mask.copy()
        
        # Aplicar seamless cloning
        try:
            result = cv2.seamlessClone(src_scaled, dst, mask_scaled, center, cv2.MIXED_CLONE)
            results.append(result)
        except:
            continue
    
    if not results:
        return dst
    
    # Promedio ponderado de resultados
    final_result = np.zeros_like(dst, dtype=np.float32)
    for result in results:
        final_result += result.astype(np.float32)
    
    final_result /= len(results)
    return final_result.astype(np.uint8)

def seamless_clone_pipeline(src_path, dst_path, output_dir, center=None, 
                          mask_method='threshold', blend_types=['normal', 'mixed'],
                          use_multi_scale=False, show_results=True):
    """
    Pipeline completo de seamless cloning mejorado
    
    Args:
        src_path: Ruta imagen fuente
        dst_path: Ruta imagen destino
        output_dir: Directorio de salida
        center: Centro donde colocar (None para auto-centrar)
        mask_method: Método para crear máscara
        blend_types: Lista de tipos de blending a probar
        use_multi_scale: Si usar blending multi-escala
        show_results: Si mostrar resultados
    """
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Leer imágenes
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    
    if src is None or dst is None:
        print("Error: No se pudieron cargar las imágenes")
        return
    
    print(f"Imagen fuente: {src.shape}")
    print(f"Imagen destino: {dst.shape}")
    
    # Crear máscara automática
    print(f"Creando máscara usando método: {mask_method}")
    src_mask = create_automatic_mask(src, method=mask_method)
    
    # Auto-centrar si no se especifica centro
    if center is None:
        center = (dst.shape[1] // 2, dst.shape[0] // 2)
        print(f"Centro automático: {center}")
    
    # Guardar máscara para inspección
    mask_path = os.path.join(output_dir, "generated_mask.jpg")
    cv2.imwrite(mask_path, src_mask)
    print(f"Máscara guardada en: {mask_path}")
    
    results = {}
    
    # Probar diferentes tipos de blending
    for blend_type in blend_types:
        print(f"\nProbando blending tipo: {blend_type}")
        
        try:
            if use_multi_scale and blend_type == 'mixed':
                output = multi_scale_blending(src, dst, src_mask, center)
                suffix = f"{blend_type}_multiscale"
            else:
                output = enhanced_seamless_clone(src, dst, src_mask, center, blend_type)
                suffix = blend_type
            
            # Guardar resultado
            base_name = os.path.splitext(os.path.basename(src_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{suffix}_CLONE.jpg")
            cv2.imwrite(output_path, output)
            
            results[blend_type] = output
            print(f"✓ Guardado: {output_path}")
            
        except Exception as e:
            print(f"✗ Error con {blend_type}: {e}")
    
    # Mostrar resultados comparativos
    if show_results and results:
        plt.figure(figsize=(15, 10))
        
        # Imagen original
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        plt.title('Fuente')
        plt.axis('off')
        
        plt.subplot(2, 3, 2) 
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.title('Destino')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(src_mask, cv2.COLOR_BGR2RGB))
        plt.title('Máscara')
        plt.axis('off')
        
        # Resultados
        for idx, (blend_type, result) in enumerate(results.items()):
            plt.subplot(2, 3, 4 + idx)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f'Resultado {blend_type}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches='tight')
        plt.show()
    
    return results

# Ejemplo de uso mejorado
if __name__ == "__main__":
    # Configuración
    src_path = "person/9999962_00000_d_0000006_18.jpg"
    dst_path = "images/0000013_00465_d_0000067.jpg"
    output_dir = "output/enhanced_results"
    
    # Pipeline completo con múltiples opciones
    results = seamless_clone_pipeline(
        src_path=src_path,
        dst_path=dst_path,
        output_dir=output_dir,
        center=(745, 693),  # o None para auto-centrar
        mask_method='threshold',  # 'threshold', 'grabcut', 'contour'
        blend_types=['normal', 'mixed'],
        use_multi_scale=False,
        show_results=True
    )
    
    print(f"\n✓ Pipeline completado. Resultados en: {output_dir}")
    print(f"✓ Se generaron {len(results)} variaciones")
    
    # Ejemplo con diferentes métodos de máscara
    for mask_method in ['threshold', 'contour']:
        method_output_dir = f"output/mask_{mask_method}"
        
        results_method = seamless_clone_pipeline(
            src_path=src_path,
            dst_path=dst_path,
            output_dir=method_output_dir,
            center=(660, 253),
            mask_method=mask_method,
            blend_types=['mixed'],
            show_results=False
        )
        
        print(f"✓ Método {mask_method} completado")