import unittest
from unittest.mock import patch, MagicMock, create_autospec
import workloads_pb2
import workloads_pb2_grpc
from video_streamer_servicer import VideoStreamerServicer

class TestVideoStreamerServicer(unittest.TestCase):

    @patch('cv2.face.LBPHFaceRecognizer_create')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.createBackgroundSubtractorMOG2')
    def setUp(self, mock_bg_subtractor, mock_cascade_classifier, mock_face_recognizer):
        self.servicer = VideoStreamerServicer()

    def test_init(self):
        self.assertIsInstance(self.servicer.recognizer, MagicMock)
        self.assertIsInstance(self.servicer.faceCascade, MagicMock)
        self.assertIsInstance(self.servicer.face_detector, MagicMock)
        self.assertIsInstance(self.servicer.bg_subtractor, MagicMock)

    @patch('cv2.imdecode')
    @patch('os.makedirs')
    def test_stream_video_face_recognition(self, mock_makedirs, mock_imdecode):
        mock_context = create_autospec(grpc.ServicerContext)
        request_iterator = iter([workloads_pb2.VideoChunk(processing_type='face_recognition', data=b'test_data')])
        mock_imdecode.return_value = MagicMock()
        response = next(self.servicer.StreamVideo(request_iterator, mock_context))
        self.assertEqual(response.message, "Stream processing completed")

    @patch('cv2.imdecode')
    @patch('os.makedirs')
    def test_stream_video_motion_detection(self, mock_makedirs, mock_imdecode):
        mock_context = create_autospec(grpc.ServicerContext)
        request_iterator = iter([workloads_pb2.VideoChunk(processing_type='motion_detection', data=b'test_data')])
        mock_imdecode.return_value = MagicMock()
        response = next(self.servicer.StreamVideo(request_iterator, mock_context))
        self.assertEqual(response.message, "Stream processing completed")

    @patch('cv2.imdecode')
    @patch('os.makedirs')
    def test_stream_video_emotion_recognition(self, mock_makedirs, mock_imdecode):
        mock_context = create_autospec(grpc.ServicerContext)
        request_iterator = iter([workloads_pb2.VideoChunk(processing_type='emotion_recognition', data=b'test_data')])
        mock_imdecode.return_value = MagicMock()
        response = next(self.servicer.StreamVideo(request_iterator, mock_context))
        self.assertEqual(response.message, "Stream processing completed")

    @patch('cv2.imdecode', side_effect=Exception('Decoding error'))
    @patch('os.makedirs')
    def test_stream_video_decoding_error(self, mock_makedirs, mock_imdecode):
        mock_context = create_autospec(grpc.ServicerContext)
        request_iterator = iter([workloads_pb2.VideoChunk(processing_type='motion_detection', data=b'test_data')])
        response = next(self.servicer.StreamVideo(request_iterator, mock_context))
        self.assertEqual(response.message, "Error decoding frame")

    @patch('cv2.cvtColor', side_effect=Exception('Processing error'))
    def test_face_recognition_error(self, mock_cvtcolor):
        frame = MagicMock()
        result = self.servicer.face_recognition(frame)
        self.assertEqual(result, "Face recognition error")

    @patch('cv2.findContours', side_effect=Exception('Processing error'))
    def test_motion_detection_error(self, mock_findContours):
        frame = MagicMock()
        result = self.servicer.motion_detection(frame)
        self.assertEqual(result, "Motion detection error")

    @patch('deepface.DeepFace.analyze', side_effect=Exception('Processing error'))
    def test_emotion_recognition_error(self, mock_analyze):
        frame = MagicMock()
        result = self.servicer.emotion_recognition(frame)
        self.assertEqual(result, "Emotion recognition error")

if __name__ == '__main__':
    unittest.main()
