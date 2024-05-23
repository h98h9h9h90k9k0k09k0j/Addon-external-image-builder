from concurrent import futures
import grpc
import time
import workloads_pb2_grpc
from video_streamer import VideoStreamerServicer
from task_manager import TaskManagerServicer

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    workloads_pb2_grpc.add_VideoStreamerServicer_to_server(VideoStreamerServicer(), server)
    workloads_pb2_grpc.add_TaskManagerServicer_to_server(TaskManagerServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
