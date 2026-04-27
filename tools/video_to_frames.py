import cv2
import os
import tkinter as tk
from tkinter import filedialog

def extract_frames():
    # Ocultar la ventana principal de tkinter
    root = tk.Tk()
    root.withdraw()

    print("Por favor selecciona el archivo de video en la ventana emergente...")

    # Abrir diálogo para seleccionar video
    video_path = filedialog.askopenfilename(
        title="Selecciona el video para extraer frames",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("Todos los archivos", "*.*")]
    )

    if not video_path:
        print("No se seleccionó ningún video.")
        return

    # Obtener el directorio del video y el nombre sin extensión
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    name_no_ext = os.path.splitext(video_name)[0]

    # Crear carpeta de salida con el mismo nombre del video
    output_dir = os.path.join(video_dir, name_no_ext)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error al crear el directorio {output_dir}: {e}")
        return

    print(f"Video seleccionado: {video_path}")
    print(f"Carpeta de salida: {output_dir}")

    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total de frames estimados: {total_frames}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Guardar frame
        # Usamos 5 dígitos para el ordenamiento (00000.jpg, 00001.jpg, etc.)
        frame_name = f"frame_{frame_count:05d}.jpg"
        output_path = os.path.join(output_dir, frame_name)
        
        cv2.imwrite(output_path, frame)
        
        frame_count += 1
        
        # Mostrar progreso cada 50 frames
        if frame_count % 50 == 0:
            print(f"Extrayendo frame {frame_count}/{total_frames if total_frames > 0 else '?'}", end='\r')

    cap.release()
    print(f"\nProceso completado.")
    print(f"Se extrajeron {frame_count} frames y se guardaron en '{output_dir}'.")

if __name__ == "__main__":
    extract_frames()
