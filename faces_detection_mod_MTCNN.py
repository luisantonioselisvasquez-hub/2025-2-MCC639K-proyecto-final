from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import numpy as np
import os
import time # LS
from concurrent.futures import ThreadPoolExecutor, as_completed  # LS
start_time = time.time() # LS
	
detector1 = MTCNN()
faces_dir = os.path.join(os.path.dirname(__file__), 'faces') # LS
images = os.listdir(faces_dir) #LS
os.makedirs('faces/mtcnn')
num_threads = 4  # LS
def detect_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return detector1.detect_faces(img_rgb)
imgs = []
img_names = []
for image in images:
    img_path = os.path.join(faces_dir, image)
    img = cv2.imread(img_path)
    print("Leyendo:", img_path, "->", "OK" if img is not None else "FALLO")
    if img is not None:
        imgs.append(img)
        img_names.append(image)
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(detect_faces, img) for img in imgs]
    results = [f.result() for f in futures]
    
    #MTCNN
for img, faces, name in zip(imgs, results, img_names):
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(os.path.join('faces', 'mtcnn', name), img)
        
    cv2.imwrite(os.path.join('faces', 'mtcnn', image), img)
    cv2.destroyAllWindows()

end_time = time.time() #LS
execution_time = end_time - start_time #LS
print(f"\nTiempo total de ejecución: {execution_time:.2f} segundos") #LS
