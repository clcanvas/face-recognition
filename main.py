import cv2
import numpy as np
import face_recognition
import os
import dlib

class faceRecog:
    discovered_locations = []
    discovered_encodings = []
    face_names = []
    known_encodings = []
    known_names = []

    INNER_MARGIN_SIDES = 25
    INNER_MARGIN_POLES = 25

    def start_recog(self):
        for image in os.listdir("faces"):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            try:
                face_encoding = face_recognition.face_encodings(face_image)[0]
                print(image)
            except:
                print(f"No face found in \"{image}\", continuing to next...")
                
            self.known_encodings.append(face_encoding)
            self.known_names.append(image)
        
        print("Retrieving video, opening first in directory")
        try:
            video = os.listdir("content")[0]
        except: 
            print("No video found. Please place a video (preferably mp4) into the content directory. Exiting...")
            return
        cap = cv2.VideoCapture(f"content/{video}")
        print(f"Opening video: {video}")

        while True: 
            ret, frame = cap.read()
            if not ret: 
                print("Video ended")
                break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 0:
                scaled_rgb = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
                
                if (dlib.DLIB_USE_CUDA):
                    self.discovered_locations = face_recognition.face_locations(scaled_rgb, model="cnn")
                else:
                    self.discovered_locations = face_recognition.face_locations(scaled_rgb, model="hog")

                self.discovered_encodings = face_recognition.face_encodings(scaled_rgb, self.discovered_locations)
                self.face_names = []
            
                for face_encoding in self.discovered_encodings:
                        matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                        
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.known_names[best_match_index]     
                            self.face_names.append(f'{name[0:name.index(".")]}')
            
            for (top, right, bottom, left), name in zip(self.discovered_locations, self.face_names):            
                self.displayLocs(name, top * 4, right * 4, bottom * 4, left * 4, frame)

            cv2.imshow(f'{video}', frame)
            waitTime = int(1000 // cap.get(cv2.CAP_PROP_FPS))
            if cv2.waitKey(waitTime)  == ord("q"):
                break

    def displayLocs(self, name, top, right, bottom, left, frame):
        cv2.rectangle(frame, (left - self.INNER_MARGIN_SIDES, top - self.INNER_MARGIN_POLES), (right + self.INNER_MARGIN_SIDES, bottom + self.INNER_MARGIN_POLES), (0, 0, 255), 2)
        cv2.rectangle(frame, (left - self.INNER_MARGIN_SIDES, bottom + 35 + self.INNER_MARGIN_POLES), (right + self.INNER_MARGIN_SIDES, bottom + self.INNER_MARGIN_POLES), (0, 0, 255), -1)
        cv2.putText(frame, name, (left - 15, bottom + 25 + self.INNER_MARGIN_POLES ), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

fr = faceRecog()
fr.start_recog()
