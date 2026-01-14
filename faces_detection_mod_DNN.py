from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import numpy as np
import os
import time # LS
start_time = time.time() # LS

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

faces_dir = os.path.join(os.path.dirname(__file__), 'faces') # LS
images = os.listdir(faces_dir) #LS
os.makedirs('faces/dnn')
scale = 1.0  # LS

for image in images:
    img_path = os.path.join(faces_dir, image) #LS
    img = cv2.imread(img_path) # LS
    print("Leyendo:", img_path, "->", "OK" if img is not None else "FALLO") # LS
    if img is None: # LS
        continue #LS

    gamma = 1.0  # cambio
    lookUpTable = np.array([((i / 255.0) ** gamma) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
    if len(img.shape) == 3:
        img = cv2.LUT(img, lookUpTable)
    else:
        img = cv2.LUT(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lookUpTable)
        
    height0, width0 = img.shape[:2]
    img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width = img_scaled.shape[:2]
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(img_scaled, (450, 450)),
                                 1.0, (450, 450), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces3 = net.forward()
    
    #OPENCV DNN
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.5:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            x = int(x / scale)
            y = int(y / scale)
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
        
    img_out = cv2.resize(img, (width0, height0), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join('faces', 'dnn', image), img2)
    cv2.destroyAllWindows()

end_time = time.time() #LS
execution_time = end_time - start_time #LS
print(f"\nTiempo total de ejecución: {execution_time:.2f} segundos") #LS
