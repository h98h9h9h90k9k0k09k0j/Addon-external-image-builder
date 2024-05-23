import logging
import time
import grpc
from concurrent import futures
import workloads_pb2
import workloads_pb2_grpc
import cv2
import numpy as np

class VideoStreamerServicer(workloads_pb2_grpc.VideoStreamerServicer):
    def StreamVideo(self, request_iterator, context):
        buffer = b""
        for chunk in request_iterator:
            buffer += chunk.data
            start = 0
            while True:
                start = buffer.find(b'\xff\xd8', start)
                end = buffer.find(b'\xff\xd9', start)
                if start != -1 and end != -1:
                    jpg = buffer[start:end+2]
                    buffer = buffer[end+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Process the frame (e.g., display, analyze, etc.)
                        logging.info("Frame processed")
                        # cv2.imshow('Frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return workloads_pb2.VideoResponse(message="Video stream processed")
                else:
                    break
        cv2.destroyAllWindows()
        return workloads_pb2.VideoResponse(message="Video stream processed")

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
