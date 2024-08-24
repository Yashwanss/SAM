import numpy as np
import face_recognition as fr
import cv2
import time

video_capture = cv2.VideoCapture(0)

Jvn_image = fr.load_image_file("Me.jpg") # Rename Your Detected Face Photo as Me.jpg And Keep Both Python File And This File In The Same Script
Jvn_face_encoding = fr.face_encodings(Jvn_image)[0]

known_face_encondings = [Jvn_face_encoding]
known_face_names = ["Jeevan"] # Change It To Your Name

"""def speak(is_detected):
    if is_detected:
        print("Jeevan's Face Was Detected")
    else:
        print("Jeevan's Face Not Detected")

"""
while True:
    ret, frame = video_capture.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    face_detected = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unable To Recognize"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            face_detected = True

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    

    cv2.imshow('Face_Recognition', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
