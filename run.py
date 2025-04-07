import gradio as gr
import rclpy
import numpy as np
import torch
from ultralytics import YOLOWorld
from transformers import pipeline
from modules.Utils import Comms, update_image, halt

# Initializations
model = YOLOWorld("yolov8m-world.pt")
model.to('cuda')
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

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
                                    streaming=True) 

            depth_image = gr.Image(
                                    label="Depth Image",
                                    type="numpy",
                                    streaming=True)
            
            button_fetch.click(update_image, inputs=[dropdown,model,comms], outputs=[output_image,depth_image])
            button_halt.click(halt, inputs=[model,comms], outputs=output_image)

    return demo

def main():
    try:
        gradio_ui = create_gradio_ui()
        
        # Start Gradio in a separate thread and spin the ROS 2 node
        from threading import Thread
        def gradio_thread():
            gradio_ui.launch()
        
        gradio_thread_instance = Thread(target=gradio_thread)
        gradio_thread_instance.start()

        # Spin the ROS 2 node to process incoming images
        rclpy.spin(comms)

    except KeyboardInterrupt:
        rclpy.shutdown()
        pass

if __name__ == "__main__":
    # Initialize
    rclpy.init()    
    comms = Comms()
    main()
