"""
Etiquetador Automático de Personas para YOLO
Detecta personas automáticamente y genera anotaciones en formato YOLO

Controles:
- ESPACIO: Procesar imagen actual automáticamente
- A: Procesar todas las imágenes automáticamente
- N: Imagen siguiente
- P: Imagen anterior
- S: Guardar anotaciones actuales
- R: Revisar detecciones (mostrar/ocultar)
- Q: Salir
"""

import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

class YOLOPersonLabeler:
    def __init__(self, images_folder=None, labels_folder=None):
        # Configurar interfaz inicial
        self.root = tk.Tk()
        self.root.withdraw()  # Ocultar ventana principal
        
        # Seleccionar carpetas si no se especifican
        if images_folder is None:
            self.images_folder = self.select_images_folder()
        else:
            self.images_folder = images_folder
            
        if labels_folder is None:
            self.labels_folder = self.select_labels_folder()
        else:
            self.labels_folder = labels_folder
            
        # Verificar que se seleccionaron carpetas
        if not self.images_folder or not self.labels_folder:
            print("Error: No se seleccionaron las carpetas necesarias")
            return
            
        self.current_image_index = 0
        self.image_files = []
        self.current_image = None
        self.current_image_path = ""
        self.display_image = None
        
        # Estado del bounding box
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_boxes = []
        
        # Configuración
        self.window_name = "YOLO Auto Person Labeler"
        self.box_color = (0, 255, 0)  # Verde
        self.text_color = (255, 255, 255)  # Blanco
        self.detection_color = (0, 0, 255)  # Rojo para detecciones automáticas
        
        # Inicializar detector HOG para personas
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Estado de visualización
        self.show_detections = True
        self.auto_detections = []
        self.processing_all = False
        
        # Crear carpeta de etiquetas si no existe
        os.makedirs(self.labels_folder, exist_ok=True)
        
        # Cargar lista de imágenes
        self.load_image_list()
        
        print("=== YOLO Person Labeler ===")
        print(f"Carpeta de imágenes: {self.images_folder}")
        print(f"Carpeta de etiquetas: {self.labels_folder}")
        print(f"Imágenes encontradas: {len(self.image_files)}")
        
    def load_image_list(self):
        """Cargar lista de archivos de imagen"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        
        for ext in extensions:
            pattern = os.path.join(self.images_folder, ext)
            self.image_files.extend(glob.glob(pattern))
            pattern_upper = os.path.join(self.images_folder, ext.upper())
            self.image_files.extend(glob.glob(pattern_upper))
        
        self.image_files.sort()
        
    def select_images_folder(self):
        """Seleccionar carpeta de imágenes mediante interfaz gráfica"""
        messagebox.showinfo("Selección de Carpeta", "Selecciona la carpeta que contiene las imágenes a etiquetar")
        folder = filedialog.askdirectory(
            title="Seleccionar carpeta de imágenes",
            initialdir=os.getcwd()
        )
        return folder if folder else None
    
    def select_labels_folder(self):
        """Seleccionar carpeta destino para las etiquetas"""
        messagebox.showinfo("Selección de Carpeta", "Selecciona la carpeta donde guardar las etiquetas YOLO")
        folder = filedialog.askdirectory(
            title="Seleccionar carpeta de etiquetas",
            initialdir=os.getcwd()
        )
        return folder if folder else None
        
    def detect_persons_auto(self):
        """Detectar personas automáticamente usando HOG"""
        if self.current_image is None:
            return []
        
        print(f"Detectando personas automáticamente en: {os.path.basename(self.current_image_path)}")
        
        # Parámetros de detección HOG
        rectangles, weights = self.hog.detectMultiScale(
            self.current_image,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
            hitThreshold=0.5,
            groupThreshold=2
        )
        
        # Filtrar detecciones por confianza y tamaño
        valid_detections = []
        min_area = 1500  # Área mínima
        min_confidence = 0.8
        
        for i, (x, y, w, h) in enumerate(rectangles):
            if i < len(weights):
                confidence = weights[i]
                area = w * h
                aspect_ratio = h / w if w > 0 else 0
                
                # Filtros de validación
                if (confidence > min_confidence and 
                    area > min_area and 
                    1.2 < aspect_ratio < 4.0):  # Proporción típica de persona
                    
                    valid_detections.append(((x, y), (x + w, y + h)))
        
        print(f"Personas detectadas: {len(valid_detections)}")
        return valid_detections
    
    def process_current_image_auto(self):
        """Procesar imagen actual automáticamente"""
        self.auto_detections = self.detect_persons_auto()
        self.current_boxes = self.auto_detections.copy()
        
        # Guardar automáticamente
        if self.current_boxes:
            self.save_annotations()
        
        self.update_display()
        return len(self.current_boxes)
    
    def process_all_images_auto(self):
        """Procesar todas las imágenes automáticamente"""
        if not self.image_files:
            print("No hay imágenes para procesar")
            return
        
        self.processing_all = True
        total_processed = 0
        total_persons = 0
        
        print(f"\n=== PROCESAMIENTO AUTOMÁTICO INICIADO ===")
        print(f"Procesando {len(self.image_files)} imágenes...")
        print("-" * 50)
        
        for i, image_path in enumerate(self.image_files):
            self.current_image_index = i
            
            # Cargar imagen
            if not self.load_current_image():
                continue
            
            # Detectar personas automáticamente
            persons_detected = self.process_current_image_auto()
            
            if persons_detected > 0:
                total_processed += 1
                total_persons += persons_detected
                
            # Mostrar progreso
            filename = os.path.basename(image_path)
            print(f"[{i+1}/{len(self.image_files)}] {filename}: {persons_detected} personas")
            
            # Actualizar display cada 5 imágenes para mostrar progreso
            if i % 5 == 0:
                self.update_display()
                cv2.waitKey(1)
        
        self.processing_all = False
        
        print("-" * 50)
        print(f"=== PROCESAMIENTO COMPLETADO ===")
        print(f"Imágenes procesadas: {total_processed}/{len(self.image_files)}")
        print(f"Total personas detectadas: {total_persons}")
        print(f"Promedio personas por imagen: {total_persons/len(self.image_files):.2f}")
        
        # Volver a la primera imagen
        self.current_image_index = 0
        self.load_current_image()
        
    def load_current_image(self):
        """Cargar imagen actual y sus anotaciones existentes"""
        if not self.image_files:
            print("No hay imágenes para procesar")
            return False
            
        self.current_image_path = self.image_files[self.current_image_index]
        self.current_image = cv2.imread(self.current_image_path)
        
        if self.current_image is None:
            print(f"Error cargando imagen: {self.current_image_path}")
            return False
        
        self.display_image = self.current_image.copy()
        
        # Cargar anotaciones existentes
        self.load_existing_annotations()
        
        # Actualizar display
        self.update_display()
        
        return True
    
    def load_existing_annotations(self):
        """Cargar anotaciones existentes si existen"""
        self.current_boxes = []
        
        image_name = Path(self.current_image_path).stem
        label_path = os.path.join(self.labels_folder, f"{image_name}.txt")
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                h, w = self.current_image.shape[:2]
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == 0:  # Solo personas (clase 0 en YOLO)
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convertir de formato YOLO a coordenadas de pixel
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            self.current_boxes.append(((x1, y1), (x2, y2)))
                            
            except Exception as e:
                print(f"Error cargando anotaciones: {e}")
    
    def save_annotations(self):
        """Guardar anotaciones en formato YOLO"""
        if not self.current_image_path or not self.current_boxes:
            return
            
        image_name = Path(self.current_image_path).stem
        label_path = os.path.join(self.labels_folder, f"{image_name}.txt")
        
        h, w = self.current_image.shape[:2]
        
        try:
            with open(label_path, 'w') as f:
                for start_point, end_point in self.current_boxes:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    
                    # Asegurar que las coordenadas estén en orden correcto
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # Convertir a formato YOLO (normalizado)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Clase 0 para persona
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")
            
            print(f"Anotaciones guardadas: {label_path} ({len(self.current_boxes)} personas)")
            
        except Exception as e:
            print(f"Error guardando anotaciones: {e}")
    
    def update_display(self):
        """Actualizar imagen mostrada con bounding boxes"""
        self.display_image = self.current_image.copy()
        
        # Dibujar bounding boxes de detecciones automáticas
        if self.show_detections:
            for i, (start_point, end_point) in enumerate(self.current_boxes):
                color = self.detection_color if hasattr(self, 'auto_detections') else self.box_color
                cv2.rectangle(self.display_image, start_point, end_point, color, 2)
                
                # Etiqueta
                label = f"Persona {i+1}"
                label_pos = (start_point[0], start_point[1] - 10)
                cv2.putText(self.display_image, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # Información en la imagen
        info_text = f"Imagen: {self.current_image_index + 1}/{len(self.image_files)} | Personas: {len(self.current_boxes)}"
        cv2.putText(self.display_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        filename = os.path.basename(self.current_image_path)
        cv2.putText(self.display_image, filename, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Estado del procesamiento
        if self.processing_all:
            cv2.putText(self.display_image, "PROCESANDO AUTOMATICAMENTE...", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Indicador de detección automática
        if hasattr(self, 'auto_detections') and self.auto_detections:
            cv2.putText(self.display_image, "AUTO-DETECTADO", (10, self.display_image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Iniciar nuevo bounding box
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finalizar bounding box
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                
                # Validar que el box tenga tamaño mínimo
                if (abs(self.end_point[0] - self.start_point[0]) > 10 and 
                    abs(self.end_point[1] - self.start_point[1]) > 10):
                    
                    self.current_boxes.append((self.start_point, self.end_point))
                    print(f"Persona agregada: {len(self.current_boxes)}")
                
                self.start_point = None
                self.end_point = None
                self.update_display()
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Cancelar bounding box actual o eliminar el último
            if self.drawing:
                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.update_display()
            elif self.current_boxes:
                removed = self.current_boxes.pop()
                print(f"Persona eliminada. Quedan: {len(self.current_boxes)}")
                self.update_display()
    
    def next_image(self):
        """Ir a la siguiente imagen"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
        else:
            print("Esta es la última imagen")
    
    def previous_image(self):
        """Ir a la imagen anterior"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        else:
            print("Esta es la primera imagen")
    
    def print_instructions(self):
        """Mostrar instrucciones"""
        print("\n=== INSTRUCCIONES - ETIQUETADOR AUTOMÁTICO ===")
        print("ESPACIO: Detectar personas automáticamente en imagen actual")
        print("A: Procesar TODAS las imágenes automáticamente")
        print("N: Imagen siguiente")
        print("P: Imagen anterior")
        print("S: Guardar anotaciones actuales")
        print("R: Mostrar/ocultar detecciones")
        print("D: Eliminar todas las anotaciones de la imagen actual")
        print("H: Mostrar estas instrucciones")
        print("Q: Salir")
        print("================================================\n")
    
    def run(self):
        """Ejecutar la aplicación"""
        if not hasattr(self, 'images_folder') or not self.images_folder:
            print("Error: No se seleccionó carpeta de imágenes")
            return
            
        if not self.image_files:
            print(f"No hay imágenes en la carpeta: {self.images_folder}")
            print("Formatos soportados: JPG, JPEG, PNG, BMP")
            return
        
        self.print_instructions()
        
        # Configurar ventana y callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Cargar primera imagen
        if not self.load_current_image():
            return
        
        # Bucle principal
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q o ESC
                break
            elif key == ord(' '):  # ESPACIO - Detectar automáticamente
                self.process_current_image_auto()
            elif key == ord('a'):  # A - Procesar todas automáticamente
                self.process_all_images_auto()
            elif key == ord('s'):  # Guardar
                self.save_annotations()
            elif key == ord('n'):  # Siguiente
                self.next_image()
            elif key == ord('p'):  # Anterior
                self.previous_image()
            elif key == ord('r'):  # Revisar - mostrar/ocultar detecciones
                self.show_detections = not self.show_detections
                self.update_display()
                print(f"Detecciones {'mostradas' if self.show_detections else 'ocultas'}")
            elif key == ord('d'):  # Eliminar todas las anotaciones
                self.current_boxes = []
                self.auto_detections = []
                print("Todas las anotaciones eliminadas")
                self.update_display()
            elif key == ord('h'):  # Ayuda
                self.print_instructions()
        
        cv2.destroyAllWindows()
        print("Etiquetado finalizado")

def main():
    """Función principal"""
    print("Inicializando YOLO Person Labeler...")
    print("Se abrirán ventanas para seleccionar carpetas...")
    
    # Crear etiquetador con selección de carpetas
    labeler = YOLOPersonLabeler()
    
    # Verificar que se inicializó correctamente
    if hasattr(labeler, 'images_folder') and labeler.images_folder:
        # Ejecutar aplicación
        labeler.run()
    else:
        print("No se pudo inicializar el etiquetador. Programa terminado.")

if __name__ == "__main__":
    main()