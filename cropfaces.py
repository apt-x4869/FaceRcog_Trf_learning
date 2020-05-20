import cv2
import os
import sys
import glob

def facecrop(image):
    cascade = cv2.CascadeClassifier('/root/MLOps/14thApril2020/haarcascade_frontalface_default.xml')

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    counter = 0
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        cv2.imshow('img',sub_face)
        # Enter to Save Images
        # If anyother key will be pressed then Image will be discarded
        if cv2.waitKey() == 120:
            pass
        else:
            cv2.imwrite(fname+"_cropped_"+str(counter)+ext, sub_face)
        cv2.destroyAllWindows()
        #cv2.imwrite(fname+"_cropped_"+str(counter)+ext, sub_face)
        counter += 1
    return

#facecrop("IMG_20200320_081856.jpg")
path = '/root/mlopsproject/Untitled Folder/Katta/*.jpg'
imgList  = glob.glob(path)
for img in imgList:
    print(img)
    facecrop(img)
    os.remove(img)
