from keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import telegram


root = tkinter.Tk()
root.withdraw()

model = load_model('face_mask_detection.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

#Capture Video
video = cv2.VideoCapture(0)

result = {0: 'With Mask', 1: 'Without Mask'}
rect_color_dict = {0: (255,0,0), 1: (0,0,255)}

while(True):
	ret, img = video.read()
	# Convert into grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



	# Detect faces
	faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
  
	# Draw rectangle around the faces and crop the faces
	for (x, y, w, h) in faces:
		faces = img[y:y + w, x:x + w]
		#cv2.imshow("face",faces)
		img_resize = cv2.resize(faces,(128,128))
		img_normalize = img_resize/255.0
		img_reshape = img_normalize.reshape((1, 128, 128, 3))
		prediction = model.predict(img_reshape)[0][0]
		if(prediction > 0.5):
			prediction = 1
		else:
			prediction = 0

		print(prediction)

		cv2.rectangle(img, (x, y), (x+w, y+h), rect_color_dict[prediction], 2)
		cv2.rectangle(img, (x, y-40), (x+w, y), rect_color_dict[prediction], -1)
		cv2.putText(img, result[prediction], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
							(0, 0, 0), 2)
		#cv2.imwrite('face.jpg', faces)

		if(prediction == 1):
			#messagebox.showwarning("Acces Denied. Please use Face mask")
			token = "Your_Token"
			chat_id = 'Your Chatid'
			bot = telegram.Bot(token=token)
			bot.sendMessage(chat_id=chat_id, text="A person without a Face mask is trying to enter the premises.")

		else:
			pass
			break


	cv2.imshow('Live Video', img)
	key=cv2.waitKey(1)

	if(key==27):
		break

cv2.destroyAllWindows()
source.release()