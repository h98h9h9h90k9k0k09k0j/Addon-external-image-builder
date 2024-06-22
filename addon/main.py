from concurrent import futures
import grpc
import time
import logging
import workloads_pb2_grpc
from video_streamer import VideoStreamerServicer
from task_manager import TaskManagerServicer

def serve():
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        video_streamer_servicer = VideoStreamerServicer()
        workloads_pb2_grpc.add_VideoStreamerServicer_to_server(video_streamer_servicer, server)
        workloads_pb2_grpc.add_TaskManagerServicer_to_server(TaskManagerServicer(video_streamer_servicer), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        logging.info("Server started successfully")

        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logging.info("Server stopping due to keyboard interrupt")
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        server.stop(0)
        logging.info("Server stopped")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
