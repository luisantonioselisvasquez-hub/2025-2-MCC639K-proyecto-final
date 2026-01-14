from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import numpy as np
import os
import time # LS
start_time = time.time() # LS
	
detector2 = dlib.get_frontal_face_detector()
faces_dir = os.path.join(os.path.dirname(__file__), 'faces') # LS
images = os.listdir(faces_dir) #LS
os.makedirs('faces/dlib')
scale = 1.5

for image in images:
    img_path = os.path.join(faces_dir, image) #LS
    img = cv2.imread(img_path) # LS
    print("Leyendo:", img_path, "->", "OK" if img is not None else "FALLO") # LS
    if img is None: # LS
        continue #LS
    gamma = 0.5  # cambio
    lookUpTable = np.array([((i / 255.0) ** gamma) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
    if len(img.shape) == 3:
        img = cv2.LUT(img, lookUpTable)
    else:
        img = cv2.LUT(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lookUpTable)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width = img.shape[:2]
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces2 = detector2(gray, 2)
    
    #DLIB    
    for result in faces2:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(img1, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(os.path.join('faces', 'dlib', image), img1)
    cv2.destroyAllWindows()

end_time = time.time() #LS
execution_time = end_time - start_time #LS
print(f"\nTiempo total de ejecución: {execution_time:.2f} segundos") #LS
