import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread
import base64
from io import BytesIO
import re

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from qwen_vl_utils import process_vision_info

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Camel-Doc-OCR-062825
MODEL_ID_M = "prithivMLmods/Camel-Doc-OCR-062825"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load ViLaSR-7B
MODEL_ID_X = "AntResearchNLP/ViLaSR"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load OCRFlux-3B
MODEL_ID_T = "ChatDOC/OCRFlux-3B"
processor_t = AutoProcessor.from_pretrained(MODEL_ID_T, trust_remote_code=True)
model_t = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_T,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load ShotVL-7B
MODEL_ID_S = "Vchitect/ShotVL-7B"
processor_s = AutoProcessor.from_pretrained(MODEL_ID_S, trust_remote_code=True)
model_s = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_S,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Helper functions for object detection
def image_to_base64(image):
    """Convert a PIL image to a base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def draw_bounding_boxes(image, bounding_boxes, outline_color="red", line_width=2):
    """Draw bounding boxes on an image."""
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)
    return image

def rescale_bounding_boxes(bounding_boxes, original_width, original_height, scaled_width=1000, scaled_height=1000):
    """Rescale bounding boxes from normalized (1000x1000) to original image dimensions."""
    x_scale = original_width / scaled_width
    y_scale = original_height / scaled_height
    rescaled_boxes = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        rescaled_box = [
            xmin * x_scale,
            ymin * y_scale,
            xmax * x_scale,
            ymax * y_scale
        ]
        rescaled_boxes.append(rescaled_box)
    return rescaled_boxes

# Default system prompt for object detection
default_system_prompt = (
    "You are a helpful assistant to detect objects in images. When asked to detect elements based on a description, "
    "you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax] with the values being scaled "
    "to 512 by 512 pixels. When there are more than one result, answer with a list of bounding boxes in the form "
    "of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]."
    "Parse only the boxes; don't write unnecessary content."
)

# Function for object detection
@spaces.GPU
def run_example(image, text_input, system_prompt):
    """Detect objects in an image and return bounding box annotations."""
    model = model_x  
    processor = processor_x

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_to_base64(image)}"},
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": text_input},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches = re.findall(pattern, str(output_text))
    parsed_boxes = [[int(num) for num in match] for match in matches]
    scaled_boxes = rescale_bounding_boxes(parsed_boxes, image.width, image.height)
    annotated_image = draw_bounding_boxes(image.copy(), scaled_boxes)
    return output_text[0], str(parsed_boxes), annotated_image

def downsample_video(video_path):
    """
    Downsample a video to evenly spaced frames, returning each as a PIL image with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generate responses using the selected model for image input.
    """
    if model_name == "Camel-Doc-OCR-062825":
        processor = processor_m
        model = model_m
    elif model_name == "ViLaSR-7B":
        processor = processor_x
        model = model_x
    elif model_name == "OCRFlux-3B":
        processor = processor_t
        model = model_t
    elif model_name == "ShotVL-7B":
        processor = processor_s
        model = model_s
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generate responses using the selected model for video input.
    """
    if model_name == "Camel-Doc-OCR-062825":
        processor = processor_m
        model = model_m
    elif model_name == "ViLaSR-7B":
        processor = processor_x
        model = model_x
    elif model_name == "OCRFlux-3B":
        processor = processor_t
        model = model_t
    elif model_name == "ShotVL-7B":
        processor = processor_s
        model = model_s
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

# Define examples for image, video, and object detection inference
image_examples = [
    ["convert this page to doc [text] precisely for markdown.", "images/1.png"],
    ["convert this page to doc [table] precisely for markdown.", "images/2.png"],
    ["explain the movie shot in detail.", "images/3.png"],   
    ["fill the correct numbers.", "images/4.png"]
]

video_examples = [
    ["explain the ad video in detail.", "videos/1.mp4"],
    ["explain the video in detail.", "videos/2.mp4"]
]

object_detection_examples = [
    ["object/1.png", "detect red and yellow cars."],
    ["object/2.png", "detect the white cat."]
]

# Added CSS to style the output area as a "Canvas"
css = """
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
.canvas-output {
    border: 2px solid #4682B4;
    border-radius: 10px;
    padding: 20px;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[Doc VLMs v2 [Localization]](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=image_examples,
                        inputs=[image_query, image_upload]
                    )
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=video_examples,
                        inputs=[video_query, video_upload]
                    )
                with gr.TabItem("Object Detection / Localization"):
                    with gr.Row():
                        with gr.Column():
                            input_img = gr.Image(label="Input Image", type="pil")
                            system_prompt = gr.Textbox(label="System Prompt", value=default_system_prompt, visible=False)
                            text_input = gr.Textbox(label="Query Input")
                            submit_btn = gr.Button(value="Submit", elem_classes="submit-btn")
                        with gr.Column():
                            model_output_text = gr.Textbox(label="Model Output Text")
                            parsed_boxes = gr.Textbox(label="Parsed Boxes")
                            annotated_image = gr.Image(label="Annotated Image")

                    gr.Examples(
                        examples=object_detection_examples,
                        inputs=[input_img, text_input],
                        outputs=[model_output_text, parsed_boxes, annotated_image],
                        fn=run_example,
                        cache_examples=True,
                    )

                    submit_btn.click(
                        fn=run_example,
                        inputs=[input_img, text_input, system_prompt],
                        outputs=[model_output_text, parsed_boxes, annotated_image]
                    )

            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)

        with gr.Column():
            with gr.Column(elem_classes="canvas-output"):
                gr.Markdown("## Result.Md")
                output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=2)
                markdown_output = gr.Markdown(label="Formatted Result (Result.Md)")

            model_choice = gr.Radio(
                choices=["Camel-Doc-OCR-062825", "OCRFlux-3B", "ShotVL-7B", "ViLaSR-7B"],
                label="Select Model",
                value="Camel-Doc-OCR-062825"
            )

            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/Doc-VLMs-v2-Localization/discussions)")
            gr.Markdown("> [Camel-Doc-OCR-062825](https://huggingface.co/prithivMLmods/Camel-Doc-OCR-062825) : camel-doc-ocr-062825 model is a fine-tuned version of qwen2.5-vl-7b-instruct, optimized for document retrieval, content extraction, and analysis recognition. built on top of the qwen2.5-vl architecture, this model enhances document comprehension capabilities.")
            gr.Markdown("> [OCRFlux-3B](https://huggingface.co/ChatDOC/OCRFlux-3B) : ocrflux-3b model that's fine-tuned from qwen2.5-vl-3b-instruct using our private document datasets and some data from olmocr-mix-0225 dataset. optimized for document retrieval, content extraction, and analysis recognition. the best way to use this model is via the ocrflux toolkit.")
            gr.Markdown("> [ViLaSR](https://huggingface.co/AntResearchNLP/ViLaSR) : vilasr-7b model as presented in reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. efficient reasoning capabilities.")
            gr.Markdown("> [ShotVL-7B](https://huggingface.co/Vchitect/ShotVL-7B) : shotvl-7b is a fine-tuned version of qwen2.5-vl-7b-instruct, trained by supervised fine-tuning on the largest and high-quality dataset for cinematic language understanding to date. it currently achieves state-of-the-art performance on shotbench.")
            gr.Markdown(">⚠️note: all the models in space are not guaranteed to perform well in video inference use cases.")  
            
    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)