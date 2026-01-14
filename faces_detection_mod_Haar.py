import cv2
import dlib
import numpy as np
import os
import time # LS
start_time = time.time() # LS
	
classifier2 = cv2.CascadeClassifier('models/haarcascade_frontalface2.xml')
faces_dir = os.path.join(os.path.dirname(__file__), 'faces') # LS
images = os.listdir(faces_dir) #LS
os.makedirs('faces/haar')
for image in images:
    img_path = os.path.join(faces_dir, image) #LS
    img = cv2.imread(img_path) # LS
    print("Leyendo:", img_path, "->", "OK" if img is not None else "FALLO") # LS
    if img is None: # LS
        continue #LS
    height, width = img.shape[:2]
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gamma = 4.0  # cambio
    lookUpTable = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8") #cambio
    img = cv2.LUT(img, lookUpTable) #cambio
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cambio
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # cambio: más ligero y local
    gray = clahe.apply(gray)
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) 
    f_scaled = classifier2.detectMultiScale(scaled, minNeighbors=3)  # cambio
    faces4 = [(int(x/2.0), int(y/2.0), int(w/2.0), int(h/2.0)) for (x, y, w, h) in f_scaled]
 
    #HAAR        
    for result in faces4:
        x, y, w, h = result
        x1, y1 = x + w, y + h
        cv2.rectangle(img3, (x, y), (x1, y1), (0, 0, 255), 2)
        
    cv2.imwrite(os.path.join('faces', 'haar', image), img3)
    cv2.destroyAllWindows()

end_time = time.time() #LS
execution_time = end_time - start_time #LS
print(f"\nTiempo total de ejecución: {execution_time:.2f} segundos") #LS
