# Sign Language Recognition Using Hand Gestures

## Introduction
This project aims to create a sign language recognition system using hand gestures. The system utilizes machine learning and image processing techniques to interpret sign language, thereby aiding communication for the deaf and hard of hearing community.

## Technologies Used
- Python
- OpenCV (for image processing)
- Pickle (for data serialization and deserialization)
- JSON (for data handling)
- Other Python libraries as listed in `requirements.txt`

## Installation
To run this project, you will need Python installed on your system. After cloning the repository, install the required dependencies:

```bash
pip install -r requirements.txt
```
# Usage

To effectively use this project, follow these steps:

1. **Collecting Images**: Run `collect_images.py` to collect images for training the model.
2. **Creating Dataset**: Use `create_dataset.py` to create the dataset from the collected images.
3. **Training the Classifier**: Execute `train_classifier.py` to train the machine learning model.
4. **Real-Time Recognition**: Run `inference_classifier.py` for real-time sign language recognition.
5. **Graphical User Interface**: Utilize `Final_GUI.py` for an interactive GUI experience.

# Files and Directories

- `Azure-ttk-theme-main`: Directory containing the theme for the GUI.
- `data.pickle`: A serialized file containing the processed data for the model.
- `labels_dict.json`: A JSON file that includes the labeling information for the dataset.
- `model.p`: The trained machine learning model file.
- `output.mp3`: An audio file used for output in the project.
- `requirements.txt`: A text file listing all necessary dependencies for the project.

# Contributing

Contributions are welcome. Please adhere to the following guidelines for contributing to this project:

1. Fork the repository and create your branch from main.
2. Ensure your code adheres to the existing style to maintain the project's consistency.
3. Update the README if you add new features or make changes that affect how the project works.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments

Special thanks to the course instructors and peers who provided invaluable feedback and support throughout the development of this project.
