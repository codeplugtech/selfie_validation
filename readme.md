# Advanced Selfie Validation API

This project is an advanced selfie validation API built using Python, FastAPI, and machine learning models like YOLO and MediaPipe. The API validates uploaded images to determine if they qualify as selfie shots based on criteria such as face detection, composition, brightness, and contrast.

## Features

- **YOLOv8 Face Detection**: Detects faces in uploaded images to validate selfies.
- **MediaPipe Face Mesh**: Analyzes face landmarks to enhance validation accuracy.
- **Selfie Scoring**: Computes a score based on face size, image brightness, contrast, and detection confidence.
- **Batch Processing**: Supports uploading and validating up to 20 images at once.
- **Detailed Validation Feedback**: Provides reasons for invalid selfies and overall statistics.

## Requirements

- Python 3.8 or higher
- FastAPI
- OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- Uvicorn (for running the FastAPI app)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/advanced-selfie-validator.git
   cd advanced-selfie-validator
   
2. create and activate virtual environment
      ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
    ```bash
   pip install -r requirements.txt

# API Endpoint
``` POST /validate_advanced_selfies/ ```

Upload images for validation.

- Request:

    - Content-Type: multipart/form-data
    - Body: List of image files (maximum 20 images).
  
- Response:

    - total_images: Total number of uploaded images.
    - valid_images: List of valid selfie images with their scores.
    - invalid_images: List of invalid images with reasons and scores.
    - average_score: Average selfie score of valid images.

### Example using curl

    curl -X POST "http://localhost:8000/validate_advanced_selfies/" \
    -F "files=@image1.jpg" \
    -F "files=@image2.jpg"

### Example Response

    {
      "total_images": 2,
      "valid_images": [
        {
          "filename": "image1.jpg",
          "valid": true,
          "score": 0.85,
          "reason": "Selfie shot"
        }
      ],
      "invalid_images": [
        {
          "filename": "image2.jpg",
          "valid": false,
          "score": 0.45,
          "reason": "Not a clear selfie shot"
        }
      ],
      "average_score": 0.85
    } 



## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

Contact

For questions or support, please contact [satz@codeplugtech.in].