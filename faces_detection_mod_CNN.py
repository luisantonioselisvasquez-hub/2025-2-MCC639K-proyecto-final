import cv2
import numpy as np
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
start_time = time.time()

# Modelo
vgg_dir = os.path.join(os.path.dirname(__file__), 'VGG')
prototxt_path = os.path.join(vgg_dir, "deploy.prototxt")
weights_path  = os.path.join(vgg_dir, "res10_300x300_ssd_iter_140000.caffemodel")
print("Verificando existencia de archivos de modelo...")
print(" prototxt:", prototxt_path, "->", os.path.exists(prototxt_path))
print(" weights :", weights_path, "->", os.path.exists(weights_path))

# Cada hilo tendrá su propia instancia de Net
thread_local = threading.local()
def get_thread_net():
    if not hasattr(thread_local, "net"):
        print(f"[thread {threading.get_ident()}] Cargando modelo en hilo...")
        thread_local.net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    return thread_local.net

# Directorios
faces_dir = os.path.join(os.path.dirname(__file__), 'faces')
images = sorted(os.listdir(faces_dir))
output_dir = os.path.join('faces', 'vgg')
os.makedirs(output_dir, exist_ok=True)
num_threads = 4
input_size = (360, 360)

# Función de detección
def detect_faces_vgg(img):
    net = get_thread_net()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, input_size),1.0,input_size,(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    confidences = []
    all_conf = [float(detections[0,0,i,2]) for i in range(detections.shape[2])
                if not np.isnan(detections[0,0,i,2]) and detections[0,0,i,2] > 0]
    if len(all_conf) == 0:
        return [], [], img
    max_conf = max(all_conf)
    dyn_thresh = max(0.2, max_conf*max_conf*max_conf*max_conf/ 1.6)
    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if np.isnan(conf) or conf < dyn_thresh:
            continue
        box = detections[0,0,i,3:7]
        if np.any(np.isnan(box)):
            continue
        box = box * np.array([w, h, w, h])
        x1 = int(max(0, min(w - 1, box[0])))
        y1 = int(max(0, min(h - 1, box[1])))
        x2 = int(max(0, min(w - 1, box[2])))
        y2 = int(max(0, min(h - 1, box[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        confidences.append(conf)
    return boxes, confidences, img

# Cargar imágenes
imgs = []
img_names = []
for image in images:
    img_path = os.path.join(faces_dir, image)
    img = cv2.imread(img_path)
    print("Leyendo:", img_path, "->", "OK" if img is not None else "FALLO")
    if img is not None:
        imgs.append(img)
        img_names.append(image)

# Procesamiento paralelo
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(detect_faces_vgg, img) for img in imgs]
    results = [f.result() for f in futures]

# Guardar resultados
for (orig, res, name) in zip(imgs, results, img_names):
    boxes, confidences, out_img = res
    out = out_img.copy()
    if len(boxes) == 0:
        print(f"[SSD-VGG] No detectó rostros en {name}")
    else:
        for (box, conf) in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0,0,255), 2)
            text = f"{conf*100:.1f}%"
            cv2.putText(out, text, (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    cv2.imwrite(os.path.join(output_dir, name), out)
print("\nDetecciones SSD+VGG16 guardadas en:", output_dir)
end_time = time.time()
print(f"\nTiempo total de ejecución: {end_time - start_time:.2f} segundos")
