# Object Detection System for Item Completeness Checking

A lightweight computer vision system that uses YOLO for real-time object detection and automated completeness checking of multipart items (for example, musical instrument components such as clarinet parts). The project is implemented in Python and uses OpenCV for image processing and CustomTkinter for a simple GUI.

## Key Features

- Real-time object detection using a YOLO model (Darknet/YOLOv5/YOLOv8 compatible).
- Automated completeness check: compares detected parts with an expected parts list and reports missing or extra items.
- Simple GUI built with CustomTkinter for ease of use and quick inspection.
- Easy-to-adapt pipeline for other multipart item inspection tasks.

## Repository Structure

- README.md - Project overview and instructions.
- src/ - Source code (detection, preprocessing, GUI).
- models/ - Trained model weights and configuration files (not included in the repo).
- data/ - Example images and annotations (if available).

> Note: File/folder names above may vary. Please review the repository tree for exact paths.

## Requirements

- Python 3.8+
- OpenCV
- torch (if using a PyTorch-based YOLO implementation)
- numpy
- customtkinter
- other dependencies as required by the chosen YOLO implementation

Install dependencies (example):

pip install -r requirements.txt

If a requirements.txt is not provided, install manually:

pip install opencv-python numpy customtkinter
# and torch + torchvision if using a PyTorch model

## Usage

1. Prepare a trained YOLO model (weights/config or a .pt file) and place it in the models/ directory.
2. Prepare any class labels or expected parts list used for completeness checking.
3. Run the detection script or launch the GUI. Example:

python src/detect.py --weights models/best.pt --source 0
# or
python src/gui_app.py

4. The system will detect parts in frames and compare them against the expected list, reporting missing items.

## Training (Optional)

If you need to train or fine-tune a model for a new item set:

1. Collect and annotate images for each part.
2. Use a YOLO training pipeline (e.g. YOLOv5 or YOLOv8) to train a model.
3. Export model weights to models/ and update the detection/config paths.

## Contributing

Contributions are welcome. If you find bugs or have feature requests, please open an issue. Pull requests should include a clear description of changes and tests where appropriate.

## License

Specify the project license here (e.g., MIT). If no license is present, add one to clarify reuse permissions.

## Contact

Maintainer: sahikjaman

For questions or collaboration, open an issue or contact the maintainer via their GitHub profile.
