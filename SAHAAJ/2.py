import cv2

face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image=cv2.imread("image.jpeg")

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(gray_image,scaleFactor=1.05,minNeighbors=5)
for x,y,w,h in faces:
    image=cv2.rectangle(image,(x+y),(x+w,y+h),(0,255,0),3)
resized=cv2.resize(image,(int(image.shape[1]/7),int(image.shape[0]/7)))
cv2.imshow("Gray",resized)
cv2.waitkey(0)
cv2.destroyaAllWindows()