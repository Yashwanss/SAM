import cv2

def detect_faces():
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read the current frame from the webcam
        _, frame = video_capture.read()

        # Convert the frame to grayscale because model can detect faces in grayscale images 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Return True 
        if len(faces) > 0:
            return True

        # if you want to see the face detection frame, uncomment
        #cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



