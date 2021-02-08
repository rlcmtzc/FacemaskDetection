from detection_utils import *
import cv2


def detect_video(input_path, output_path, frame_rate = 30):
    video_capture = cv2.VideoCapture(input_path)
    count=1

    success, frame = video_capture.read()
    if not success:
        raise ValueError("could not read video")
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(output_path, 0, frame_rate, (width,height))

     
    detected_frame = detect_images(net, classes,  frame, "", create_result_txt=False, files=True)
    #print("detect frame 0")
    video_writer.write(detected_frame)
    while success:
        count = count + 1
        success,frame = video_capture.read()
        if success:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
            print(f"detect frame {count}")
            detected_frame = detect_images(net, classes, frame, "", create_result_txt=False, files=True)
            video_writer.write(detected_frame)
    cv2.destroyAllWindows()
    video_capture.release()
    video_writer.release()

net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')
    #net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'output.pbtxt')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
detect_video("<input video path>", "<output video path>", frame_rate=30)
os.kill (os.getpid (), signal.SIGTERM)