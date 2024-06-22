import grpc
import workloads_pb2
import workloads_pb2_grpc
import logging

class TaskManagerServicer(workloads_pb2_grpc.TaskManagerServicer):
    def __init__(self, video_streamer_servicer):
        self.video_streamer_servicer = video_streamer_servicer

    def SendTask(self, request, context):
        task_id = request.task_id
        task_type = request.task_type
        payload = request.payload
        
        if task_type == "retrieve_frames":
            return self.retrieve_frames()

        # Handle other task types
        print(f"Received task: {task_id}, type: {task_type}, payload: {payload}")
        return workloads_pb2.TaskResponse(message="Task completed", task_id=task_id)

    def RetrieveFrames(self, request, context):
        frames = self.video_streamer_servicer.processed_frames
        frame_responses = [workloads_pb2.FrameData(image=frame_data.image, timestamp=frame_data.timestamp) for frame_data in frames]
        return workloads_pb2.FrameResponse(frames=frame_responses)
