import datetime
import logging
import os
import cv2
import numpy as np
import threading
import queue
from PIL import Image
import workloads_pb2
import workloads_pb2_grpc
from deepface import DeepFace

class VideoStreamerServicer(workloads_pb2_grpc.VideoStreamerServicer):
    def __init__(self):
        try:
            self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.current_path = ""
            self.count = 0
            self.lock = threading.Lock()
            self.frame_queue = queue.Queue(maxsize=30)
            self.processing = False
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_database_path = 'dataset'
            os.makedirs(self.face_database_path, exist_ok=True)
            self.trainer_file_path = os.path.join(self.face_database_path, 'trainer.yml')
            self.motion_detection_path = "img_motion_det"
            os.makedirs(self.motion_detection_path, exist_ok=True)
            self.max_saved_images = 50
            self.recognizer_trained = False
            self.processed_frames = []
            self.train_recognizer()
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise

    def StreamVideo(self, request_iterator, context):
        try:
            buffer = b""
            first_chunk = next(request_iterator)
            processing_type = first_chunk.processing_type
            buffer += first_chunk.data

            for chunk in request_iterator:
                buffer += chunk.data
                if chunk.processing_type != processing_type:
                    processing_type = chunk.processing_type
                start = 0
                while True:
                    start = buffer.find(b'\xff\xd8', start)
                    end = buffer.find(b'\xff\xd9', start)
                    if start != -1 and end != -1:
                        jpg = buffer[start:end+2]
                        buffer = buffer[end+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.frame_queue.put((frame, processing_type))
                            if not self.processing:
                                self.processing = True
                                threading.Thread(target=self.process_frames).start()
                        else:
                            break
                    else:
                        break
            return workloads_pb2.TaskResponse(message="Stream processing completed", task_id="")
        except Exception as e:
            logging.error(f"Error in StreamVideo method: {e}")
            return workloads_pb2.TaskResponse(message="StreamVideo method error", task_id="")

    def process_frames(self):
        while not self.frame_queue.empty():
            frame, processing_type = self.frame_queue.get()
            result_message = ""
            if processing_type == 'face_recognition':
                result_message = self.face_recognition(frame)
            elif processing_type == 'motion_detection':
                result_message = self.motion_detection(frame)
            elif processing_type == 'emotion_recognition':
                result_message = self.emotion_recognition(frame)

            if result_message:
                logging.info(f"Processed frame result: {result_message}")
                # Save processed frame for retrieval
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = workloads_pb2.FrameData(image=buffer.tobytes(), timestamp=datetime.datetime.now().isoformat())
                self.processed_frames.append(frame_data)
                # Remove old frames if necessary
                if len(self.processed_frames) > self.max_saved_images:
                    self.processed_frames.pop(0)
        self.processing = False

    def face_recognition(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if self.recognizer_trained:
                    id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                    if confidence < 50:
                        confidence_text = f"{round(100 - confidence)}%"
                        name = f"User {id}"
                    else:
                        confidence_text = "0%"
                        name = "unknown"

                    cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

                    if name == "unknown":
                        with self.lock:
                            self.count += 1
                            face_id = self.count
                            user_dir = f"{self.face_database_path}/User_{face_id}"
                            os.makedirs(user_dir, exist_ok=True)
                            for i in range(5):
                                face_path = f"{user_dir}/User.{face_id}.{i + 1}.jpg"
                                cv2.imwrite(face_path, gray[y:y + h, x:x + w])
                            self.train_recognizer()
                            return f"New face detected and saved as {face_path}"
                else:
                    with self.lock:
                        self.count += 1
                        face_id = self.count
                        user_dir = f"{self.face_database_path}/User_{face_id}"
                        os.makedirs(user_dir, exist_ok=True)
                        for i in range(5):
                            face_path = f"{user_dir}/User.{face_id}.{i + 1}.jpg"
                            cv2.imwrite(face_path, gray[y:y + h, x:x + w])
                        self.train_recognizer()
                        return f"New face detected and saved as {face_path}"
            return f"Face recognition completed: saw user {name} with confidence {confidence_text}"
        except cv2.error as e:
            logging.error(f"OpenCV error in face_recognition method: {e}")
            return "Face recognition error"
        except Exception as e:
            logging.error(f"Error in face_recognition method: {e}")
            return "Face recognition error"

    def train_recognizer(self):
        try:
            face_samples, ids = self.get_images_and_labels(self.face_database_path)
            if len(face_samples) > 0:
                self.recognizer.train(face_samples, np.array(ids))
                self.recognizer.write(self.trainer_file_path)
                self.recognizer_trained = True
                logging.info(f"Trained recognizer with {len(np.unique(ids))} unique faces.")
            else:
                logging.info("No faces found to train the recognizer.")
                self.recognizer_trained = False
        except Exception as e:
            logging.error(f"Error training recognizer: {e}")

    def get_images_and_labels(self, path):
        face_samples = []
        ids = []
        for user_dir in os.listdir(path):
            user_path = os.path.join(path, user_dir)
            if os.path.isdir(user_path):
                for image_name in os.listdir(user_path):
                    if image_name.endswith('.jpg'):
                        image_path = os.path.join(user_path, image_name)
                        pil_img = Image.open(image_path).convert('L')
                        img_numpy = np.array(pil_img, 'uint8')
                        face_id = int(user_dir.split('_')[-1])
                        faces = self.faceCascade.detectMultiScale(img_numpy)
                        for (x, y, w, h) in faces:
                            face_samples.append(img_numpy[y:y + h, x:x + w])
                            ids.append(face_id)
        return face_samples, ids

    def motion_detection(self, frame):
        try:
            date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fgmask = self.bg_subtractor.apply(frame)
            _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    motion_detected = True

            with self.lock:
                if motion_detected:
                    date_path = os.path.join(self.motion_detection_path, datetime.datetime.now().strftime("%Y-%m-%d"))
                    os.makedirs(date_path, exist_ok=True)
                    image_path = os.path.join(date_path, f"frame_{date_time}.jpg")
                    cv2.imwrite(image_path, frame)
                    self.cleanup_old_images(date_path)
                    return f"Motion detected and saved as {image_path}"
                else:
                    return "No significant motion detected"
        except Exception as e:
            logging.error(f"Error in motion_detection method: {e}")
            return "Motion detection error"

    def cleanup_old_images(self, directory):
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
        if len(image_files) > self.max_saved_images:
            image_files.sort(key=os.path.getctime)
            for image_file in image_files[:len(image_files) - self.max_saved_images]:
                os.remove(image_file)
                logging.info(f"Removed old image: {image_file}")

    def emotion_recognition(self, frame):
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            faces = self.faceCascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                message = f'Emotion "{emotion}" detected'
                return message
        except Exception as e:
            logging.error(f"Error in emotion_recognition method: {e}")
            return "Emotion recognition error"
