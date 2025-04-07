"""
This was the original implementation of the code, which was later modified to make it cleaner
"""


import gradio as gr
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import time
import torch
from ultralytics import YOLOWorld
import cv2
import os
from transformers import pipeline
from PIL import Image, ImageDraw

last_valid_image = None
model = YOLOWorld("yolov8m-world.pt")
model.to('cuda')
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
go = 0

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Create CvBridge to convert ROS image messages
        self.br = CvBridge()

        # Create the ROS 2 subscription to listen to compressed images
        self.subscription = self.create_subscription(
            CompressedImage,
            'video_frames',  # Replace with the correct topic name
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
    def listener_callback(self, msg):
        global last_valid_image

        try:
            # Decompress the image (ROS message -> OpenCV image)
            np_array = np.frombuffer(msg.data, np.uint8)  # Convert byte data to numpy array
            self.current_frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            self.current_frame = cv2.rotate(self.current_frame, cv2.ROTATE_90_CLOCKWISE)

            if self.current_frame is None:
                self.get_logger().error("Failed to decode image")
                return

            # Update the last valid image
            last_valid_image = self.current_frame

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def update_image(target):
    global go
    go = 1
    while go == 1:
        if last_valid_image is not None:
            # Run YOLO model inference on the last valid image
            model.set_classes([target])
            results = model.predict(last_valid_image)
            img_with_overlay = results[0].cpu().numpy()
            results[0].save(os.path.expanduser("~/remote_ws/rcvd_img/inference.jpg"))

            detected = False
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            names = results[0].names

            image = Image.fromarray(last_valid_image)
            depth = pipe(image)["depth"]

            draw = ImageDraw.Draw(depth)

            # Iterate over each bounding box and draw it
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers (if they are not already)
                targetbox = depth.crop((x1, y1, x2, y2))
                objectdepth = np.mean(np.array(targetbox))
                # Draw a rectangle (bounding box) on the image
                draw.rectangle([x1, y1, x2, y2], outline="white", width=3)

            depth.save(os.path.expanduser("~/remote_ws/rcvd_img/depth.jpg"), 'JPEG')

            # Convert the image to a NumPy array
            image_array = np.array(image)
            # Get the threshold row (safe distance, observable nearby floor)
            thres_index = image_array.shape[0] - 100
            thres_pixels_max = np.max(image_array[thres_index])
            # If any object in the center row is at the same distance with the threshold row, stop!
            
            print("GO STATUS ", go)


            yield gr.update(value=os.path.expanduser("~/remote_ws/rcvd_img/inference.jpg")) ,gr.update(value=os.path.expanduser("~/remote_ws/rcvd_img/depth.jpg"))

            # The result is a list of Results objects, we need to access the first item
            for box, conf, class_id in zip(boxes, conf, classes):
                if names[int(class_id)] == target:
                    detected = True
                    print("THRESHOLD", thres_pixels_max, "OBJECT", objectdepth)
                    if abs(objectdepth) > 0 and abs(objectdepth) < 60:
                        twist_msg = Twist()
                        x_center = (box[0] + box[2]) // 2
                        twist_msg.linear.x = 0.2
                        twist_msg.angular.z = 0.12 if x_center < image_subscriber.current_frame.shape[1] // 2 else -0.12
                        image_subscriber.publisher.publish(twist_msg)
                        break  

                    if abs(objectdepth) >= 60 and abs(objectdepth) < 120:
                        twist_msg = Twist()
                        x_center = (box[0] + box[2]) // 2
                        twist_msg.angular.z = 0.08 if x_center < image_subscriber.current_frame.shape[1] // 2 else -0.08
                        twist_msg.linear.x = 0.05
                        image_subscriber.publisher.publish(twist_msg)
                        print("SUSPECTED OBSTRUCTION, SLOWING DOWN!") 
                        print("GO STATUS ", go)
                        break

                    if abs(objectdepth) >= 120:
                        twist_msg = Twist()
                        twist_msg.angular.z = 0.0
                        twist_msg.linear.x = 0.0
                        image_subscriber.publisher.publish(twist_msg)
                        go = 0
                        print("OBSTRUCTION DETECTED, STOPPING!") 
                        print("GO STATUS ", go)
                        break

            if not detected:
                twist_msg = Twist()
                twist_msg.angular.z = 0.15
                image_subscriber.publisher.publish(twist_msg)

            time.sleep(0.1)

def depth_display():
    while True:
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        image = Image.fromarray(last_valid_image)
        depth = pipe(image)["depth"]
        depth.save(os.path.expanduser("~/remote_ws/rcvd_img/depth.jpg"), 'JPEG')
        yield gr.update(value=os.path.expanduser("~/remote_ws/rcvd_img/depth.jpg"))
        time.sleep(0.1) 

def halt():
    global go
    go = 0
    while go ==0:
        if last_valid_image is not None:
            # Run YOLO model inference on the last valid image
            results = model.predict(last_valid_image)
            img_with_overlay = results[0].cpu().numpy()
            results[0].save(os.path.expanduser("~/remote_ws/rcvd_img/inference.jpg"))

            twist_msg = Twist()
            twist_msg.angular.z = 0.0
            twist_msg.linear.x = 0.0
            image_subscriber.publisher.publish(twist_msg)

            # Update Gradio with the processed image
            yield gr.update(value=os.path.expanduser("~/remote_ws/rcvd_img/inference.jpg"))  # Show the image in Gradio
            time.sleep(0.15)

def create_gradio_ui():

    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>🐢 TurtleBert RasPi Cam + YOLOWorld + Depth-Anything V2 🐢 </h1>")

        with gr.Row():
            dropdown = gr.Textbox(
                        label="What to Hunt?"
                    )

        with gr.Row():
            button_fetch = gr.Button("Start Hunting")
            button_halt = gr.Button("Halt Robot")

        with gr.Row():
            output_image = gr.Image(
                                    label="TurtleBert sees...",
                                    type="numpy",
                                    streaming=True)  # Image type for Gradio to display as PIL

            depth_image = gr.Image(
                                    label="Depth Image",
                                    type="numpy",
                                    streaming=True)
            
            button_fetch.click(update_image, inputs=dropdown, outputs=[output_image,depth_image])
            button_halt.click(halt, outputs=output_image)


    return demo

# Initialize ROS2
rclpy.init()    
image_subscriber = ImageSubscriber()

def main():
    try:
        gradio_ui = create_gradio_ui()
        
        # Start Gradio in a separate thread and spin the ROS 2 node
        from threading import Thread
        def gradio_thread():
            gradio_ui.launch()
        
        gradio_thread_instance = Thread(target=gradio_thread)
        gradio_thread_instance.start()
        rclpy.spin(image_subscriber)

    except KeyboardInterrupt:
        image_subscriber.destroy_node()
        rclpy.shutdown()
        pass

if __name__ == "__main__":
    main()