
import os
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

last_valid_image = None
go = 0

class Comms(Node):
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

def update_image(target,model,comms):
    global go
    go = 1
    while go == 1:
        if last_valid_image is not None:
            # Run YOLO model inference on the last valid image
            model.set_classes([target])
            results = model.predict(last_valid_image)
            img_with_overlay = results[0].cpu().numpy()
            results[0].save(os.path.expanduser("<SAVE IMAGE TARGET PATH>"))

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

            depth.save(os.path.expanduser("<SAVE IMAGE TARGET PATH>"), 'JPEG')

            # Convert the image to a NumPy array
            image_array = np.array(image)
            # Get the threshold row (safe distance, observable nearby floor)
            thres_index = image_array.shape[0] - 100
            thres_pixels_max = np.max(image_array[thres_index])
            # If any object in the center row is at the same distance with the threshold row, stop!
            
            print("GO STATUS ", go)

            yield gr.update(value=os.path.expanduser("<SAVED IMAGE SOURCE PATH>")) ,gr.update(value=os.path.expanduser("<SAVED IMAGE SOURCE PATH>"))

            # The result is a list of Results objects, we need to access the first item
            for box, conf, class_id in zip(boxes, conf, classes):
                if names[int(class_id)] == target:
                    detected = True
                    print("THRESHOLD", thres_pixels_max, "OBJECT", objectdepth)
                    if abs(objectdepth) > 0 and abs(objectdepth) < 60:
                        twist_msg = Twist()
                        x_center = (box[0] + box[2]) // 2
                        twist_msg.linear.x = 0.2
                        twist_msg.angular.z = 0.12 if x_center < comms.current_frame.shape[1] // 2 else -0.12
                        comms.publisher.publish(twist_msg)
                        break  

                    if abs(objectdepth) >= 60 and abs(objectdepth) < 120:
                        twist_msg = Twist()
                        x_center = (box[0] + box[2]) // 2
                        twist_msg.angular.z = 0.08 if x_center < comms.current_frame.shape[1] // 2 else -0.08
                        twist_msg.linear.x = 0.05
                        comms.publisher.publish(twist_msg)
                        print("SUSPECTED OBSTRUCTION, SLOWING DOWN!") 
                        print("GO STATUS ", go)
                        break

                    if abs(objectdepth) >= 120:
                        twist_msg = Twist()
                        twist_msg.angular.z = 0.0
                        twist_msg.linear.x = 0.0
                        comms.publisher.publish(twist_msg)
                        go = 0
                        print("OBSTRUCTION DETECTED, STOPPING!") 
                        print("GO STATUS ", go)
                        break

            if not detected:
                twist_msg = Twist()
                twist_msg.angular.z = 0.15
                comms.publisher.publish(twist_msg)

            time.sleep(0.1)

def depth_display():
    global pipe
    while True:
        image = Image.fromarray(last_valid_image)
        depth = pipe(image)["depth"]
        depth.save(os.path.expanduser("<SAVE IMAGE TARGET PATH>"), 'JPEG')
        yield gr.update(value=os.path.expanduser("<SAVED IMAGE SOURCE PATH>"))
        time.sleep(0.1) 

def halt(model,comms):
    global go
    go = 0
    while go ==0:
        if last_valid_image is not None:
            # Run YOLO model inference on the last valid image
            results = model.predict(last_valid_image)
            img_with_overlay = results[0].cpu().numpy()
            results[0].save(os.path.expanduser("<SAVE RESULTS TARGET PATH>"))

            twist_msg = Twist()
            twist_msg.angular.z = 0.0
            twist_msg.linear.x = 0.0
            comms.publisher.publish(twist_msg)

            # Update Gradio with the processed image
            yield gr.update(value=os.path.expanduser("<SAVED IMAGE SOURCE PATH>"))  # Show the image in Gradio

            time.sleep(0.15)
