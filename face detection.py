import cv2
import sys


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)#function to read from vedio cam
#to give the stored vedio give name of the vedio instead of 0 i.e "vedio.mp4".

while True:
    # Capture frame-by-frame
    #infinite loop
    ret, frame = video_capture.read()#vedio_capture stores all the images and .read eill read from vedio capture

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convertion from color to gray

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
        
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        eyes = eye_cascade.detectMultiScale(gray)#detects the eyes in the pic using only the face
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
