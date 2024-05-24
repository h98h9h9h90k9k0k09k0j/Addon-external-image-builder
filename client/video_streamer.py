import datetime
import logging
import os
import time
import grpc
from concurrent import futures
import workloads_pb2
import workloads_pb2_grpc
import cv2
import numpy as np
from deepface import DeepFace

class VideoStreamerServicer(workloads_pb2_grpc.VideoStreamerServicer):
    def __init__(self):
        # Initialize the face recognizer and other necessary components
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer/trainer.yml')
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.current_path = ""
        self.count = 0

    def StreamVideo(self, request_iterator, context):
        buffer = b""
        # Extract the processing type from the first chunk of the request
        first_chunk = next(request_iterator)
        processing_type = first_chunk.processing_type
        buffer += first_chunk.data

        if processing_type == 'face_recognition':
            # Load the face recognizer model
            self.recognizer.read('trainer/trainer.yml')
            self.count = 0
        elif processing_type == 'motion_detection':
            self.current_path = "img_motion_det" #  Please, Check if path is created inside client folder
            os.makedirs(self.current_path, exist_ok=True)
            self.count = 0
        elif processing_type == 'emotion_recognition':
            pass

        for chunk in request_iterator:
            buffer += chunk.data
            start = 0
            while True:
                try:
                    start = buffer.find(b'\xff\xd8', start)
                    end = buffer.find(b'\xff\xd9', start)
                    if start != -1 and end != -1:
                        jpg = buffer[start:end+2]
                        buffer = buffer[end+2:]
                        try:
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                if processing_type == 'face_recognition':
                                    result_message = self.face_recognition(frame)
                                elif processing_type == 'motion_detection':
                                    result_message = self.motion_detection(frame)
                                elif processing_type == 'emotion_recognition':
                                    result_message = self.emotion_recognition(frame)

                                if result_message:
                                    yield workloads_pb2.VideoResponse(message=result_message)
                        except Exception as e:
                            logging.error(f"Error decoding frame: {str(e)}")
                            # print error message
                            print(f"Error decoding frame: {str(e)}")
                            return workloads_pb2.VideoResponse(message="Error decoding frame")
                    else:
                        break
                except Exception as e:
                    logging.error(f"Error finding start and end markers: {str(e)}")
                    print(f"Error finding start and end markers: {str(e)}")
                    return workloads_pb2.VideoResponse(message="Error finding start and end markers")
        return workloads_pb2.VideoResponse(message="Stream processing completed")
    
    def face_recognition(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            self.count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User.face." + str(self.count) + ".jpg", gray[y:y+h,x:x+w])
            return f"Face detected and saved as User.face.{self.count}.jpg"
    


    def motion_detection(self, frame):
        date_time = str(datetime.datetime.now())
        fgmask = self.bg_subtractor.apply(frame) # apply background subtraction -> foreground mask image
        _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY[1]) #colors above 200 turns white. [1] = thresholded image
        contours, hierarchy = cv2. findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL = external contours, CHAIN... = compress to save memory
            
        motion_detected = False  
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #rectangle drawn around object if area is big enough
                motion_detected = True
        
        if motion_detected and self.count < 10:
            image_path = os.path.join(self.current_path, f"frame_{date_time}_{self.count}.jpg")
            cv2.imwrite(image_path, frame)
            self.count += 1
            return f"Motion detected and saved as frame_{date_time}_{self.count}.jpg"
        else:
            self.count = 0
            time.sleep(3)  # HEAVY ON THE CAMERA RENDERING!!!
        cv2.putText(frame, date_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
        return None 

    def emotion_recognition(self, frame):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = self.faceCascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            message = f'Emotion "{emotion}" detected'
            return message
        return None


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    workloads_pb2_grpc.add_VideoStreamerServicer_to_server(VideoStreamerServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
