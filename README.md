# **Doc-VLMs-v2-Localization**

A comprehensive multi-modal AI application that combines document analysis, optical character recognition (OCR), video understanding, and object detection capabilities using state-of-the-art vision-language models.

## Features

### Core Capabilities
- **Document Analysis**: Extract and convert document content to structured formats (text, tables, markdown)
- **OCR Processing**: Advanced optical character recognition for various document types
- **Video Understanding**: Analyze and describe video content with temporal awareness
- **Object Detection**: Locate and identify objects in images with bounding box annotations
- **Multi-Model Support**: Choose from four specialized vision-language models

### Supported Models
1. **Camel-Doc-OCR-062825**: Fine-tuned Qwen2.5-VL model optimized for document retrieval and content extraction
2. **OCRFlux-3B**: Specialized 3B parameter model for OCR tasks with high accuracy
3. **ViLaSR-7B**: Advanced spatial reasoning model with visual drawing capabilities
4. **ShotVL-7B**: Cinematic language understanding model trained on high-quality video datasets

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/PRITHIVSAKTHIUR/Doc-VLMs-v2-Localization.git
cd Doc-VLMs-v2-Localization

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```
gradio
spaces
torch
numpy
pillow
opencv-python
transformers
qwen-vl-utils
```

## Usage

### Running the Application
```bash
python app.py
```

The application will launch a Gradio interface accessible through your web browser.

### Interface Overview

#### Image Inference Tab
- Upload images for analysis
- Query the model with natural language
- Get structured outputs including text extraction and document conversion

#### Video Inference Tab
- Upload video files for analysis
- Generate detailed descriptions of video content
- Temporal understanding with frame-by-frame analysis

#### Object Detection Tab
- Upload images for object localization
- Specify objects to detect using natural language
- View annotated images with bounding boxes
- Get precise coordinate information

### Advanced Configuration
- **Max New Tokens**: Control response length (1-2048)
- **Temperature**: Adjust creativity/randomness (0.1-4.0)
- **Top-p**: Nuclear sampling parameter (0.05-1.0)
- **Top-k**: Token selection threshold (1-1000)
- **Repetition Penalty**: Prevent repetitive outputs (1.0-2.0)

## API Reference

### Core Functions

#### `generate_image(model_name, text, image, **kwargs)`
Process image inputs with selected model
- **Parameters**: Model selection, query text, PIL image, generation parameters
- **Returns**: Streaming text response

#### `generate_video(model_name, text, video_path, **kwargs)`
Analyze video content with temporal understanding
- **Parameters**: Model selection, query text, video file path, generation parameters
- **Returns**: Streaming analysis response

#### `run_example(image, text_input, system_prompt)`
Perform object detection with bounding box output
- **Parameters**: PIL image, detection query, system prompt
- **Returns**: Detection results, parsed coordinates, annotated image

### Helper Functions

#### `downsample_video(video_path)`
Extract representative frames from video files
- **Parameters**: Path to video file
- **Returns**: List of PIL images with timestamps

#### `rescale_bounding_boxes(boxes, width, height)`
Convert normalized coordinates to image dimensions
- **Parameters**: Bounding box coordinates, image dimensions
- **Returns**: Rescaled coordinate arrays

## Model Information

### Camel-Doc-OCR-062825
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Specialization**: Document comprehension and OCR
- **Use Cases**: Text extraction, table conversion, document analysis

### OCRFlux-3B
- **Base Model**: Qwen2.5-VL-3B-Instruct
- **Specialization**: Optical character recognition
- **Use Cases**: Text recognition, document digitization

### ViLaSR-7B
- **Base Model**: Advanced spatial reasoning architecture
- **Specialization**: Visual drawing and spatial understanding
- **Use Cases**: Complex visual reasoning tasks

### ShotVL-7B
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Specialization**: Cinematic content understanding
- **Use Cases**: Video analysis, shot detection, narrative understanding

## Examples

### Document Analysis
```python
# Query: "convert this page to doc [text] precisely for markdown"
# Input: Document image
# Output: Structured markdown format
```

### Object Detection
```python
# Query: "detect red and yellow cars"
# Input: Street scene image
# Output: Bounding boxes around detected vehicles
```

### Video Understanding
```python
# Query: "explain the ad video in detail"
# Input: Advertisement video file
# Output: Comprehensive video content analysis
```

## Performance Notes

- GPU acceleration recommended for optimal performance
- Video processing involves frame sampling (10 frames per video)
- Object detection uses normalized 512x512 coordinate system
- Streaming responses provide real-time feedback

## Limitations

- Video inference performance may vary across models
- GPU memory requirements scale with model size
- Processing time depends on input complexity and hardware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Built on Hugging Face Transformers
- Powered by Gradio for the web interface
- Utilizes Qwen vision-language model architecture
- Integrated with Spaces GPU acceleration

## Support

For issues and questions:
- Open an issue on GitHub
- Check the Hugging Face model documentation
- Review the examples and documentation
