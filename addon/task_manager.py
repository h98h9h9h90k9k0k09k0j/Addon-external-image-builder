import time
import grpc
from concurrent import futures
import workloads_pb2
import workloads_pb2_grpc

class TaskManagerServicer(workloads_pb2_grpc.TaskManagerServicer):
    def SendTask(self, request, context):
        # Handle the task request
        task_id = request.task_id
        task_type = request.task_type
        payload = request.payload
        # Process the task based on type and payload
        print(f"Received task: {task_id}, type: {task_type}, payload: {payload}")
        return workloads_pb2.TaskResponse(message="Task completed", task_id=task_id)

    def StreamTask(self, request_iterator, context):
        task_id = None
        for chunk in request_iterator:
            # Process each chunk of task data
            data = chunk.data
            task_id = chunk.task_id
            print(f"Received task chunk for task_id: {task_id}, data size: {len(data)}")
            # Perform processing on each chunk
        return workloads_pb2.TaskResponse(message="Streamed task completed", task_id=task_id)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
