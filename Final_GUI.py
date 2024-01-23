import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from threading import Thread, Event
from gtts import gTTS
import os
import time
import json
import ttkthemes as ttkM
from PIL import Image, ImageTk
import shutil

# Load your hand gesture recognition model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Load or initialize labels_dict
labels_dict_file = "./labels_dict.json"

def load_labels_dict():
    global labels_dict
    if os.path.exists(labels_dict_file):
        with open(labels_dict_file, 'r') as json_file:
            labels_dict = json.load(json_file)
    else:
        labels_dict = {}

# Call the function to load labels_dict at the beginning
load_labels_dict()

# Initialize OpenCV camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

# Data storage for gestures
stored_gestures = []

# Global variables
results = None
prediction = None

# Flag to control the main loop
running = True

# Global flag to control the processing loop
processing_enabled = True

# Event to pause and resume video processing
pause_event = Event()

# Variables for recording logic
recording = False
start_time = 0
captured_inputs = []
current_gesture = None
last_gesture = None
last_prediction_time = 0

# Tkinter GUI setup
root = tk.Tk()

# Simply set the theme
root.tk.call("source", "Azure-ttk-theme-main/azure.tcl")
root.tk.call("set_theme", "dark")

root.title("Hand Gesture Recognition GUI")
root.geometry("450x800") 

# Tkinter label to display real-time predicted inputs during recording
predicted_label = tk.Label(root, text="Predicted Inputs: ")
predicted_label.pack(pady=10)

predicted_textbox = tk.Text(root, height=4, width=40, wrap=tk.WORD)
predicted_textbox.pack(pady=10, expand=True, fill=tk.BOTH)

# Create a frame to contain the gesture-related buttons
gesture_frame = ttk.Frame(root)

capture_duration_same_prediction = 3

def set_timer():
    global capture_duration_same_prediction
    new_duration = simpledialog.askfloat("Set Timer", "Enter new capture duration (in seconds):", initialvalue=capture_duration_same_prediction)
    if new_duration is not None:
        capture_duration_same_prediction = max(0, new_duration)


# Button for Model Calibration
def calibrate_model():
    os.system("python create_dataset.py")
    os.system("python train_classifier.py")

# Button for Start Recording
def start_recording():
    global recording, start_time, captured_inputs, current_gesture, last_gesture, last_prediction_time
    recording = True
    start_time = time.time()  # Update the start time
    captured_inputs = []
    current_gesture = None
    global last_gesture, last_prediction_time  # Declare last_gesture and last_prediction_time as global variables
    last_gesture = None
    last_prediction_time = time.time()  # Set the initial time for last prediction

# Button for Stop Recording
def stop_recording():
    global recording, captured_inputs
    recording = False
    messagebox.showinfo("Recording Stopped", f"Captured Inputs: {captured_inputs}")

# Text to Speech button
def text_to_speech_realtime():
    global captured_inputs
    if captured_inputs:
        sentence = ' '.join(captured_inputs)
        tts = gTTS(sentence, lang='en')
        tts.save("output.mp3")
        os.system("start output.mp3")  # This will play the generated audio file

# Button for Reset Dataset Labels
def reset_dataset_labels():
    global labels_dict
    labels_dict = {}  # Reset labels_dict

    dataset_path = "./data"
    dataset_count = sum(os.path.isdir(os.path.join(dataset_path, f)) for f in os.listdir(dataset_path))

    for i in range(dataset_count):
        gesture_name = simpledialog.askstring("Dataset Calibration", f"Enter name for Gesture {i}:")
        labels_dict[i] = gesture_name

    with open(labels_dict_file, 'w') as json_file:
        json.dump(labels_dict, json_file)
    messagebox.showinfo("Dataset Labels Reset", "Gesture names have been reset.")

# Button for Add Gesture
def add_gesture():
    global processing_enabled, pause_event, labels_dict

    # Pause the processing loop
    pause_event.set()

    os.system("python collect_images.py")

    # Find the most recent dataset folder
    dataset_path = "./data"
    dataset_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    most_recent_dataset = max(dataset_folders, key=lambda x: int(x))

    # Calculate the new gesture label
    new_gesture_label = str(len(labels_dict))

    # Ask the user for the new gesture name
    new_gesture_name = simpledialog.askstring("Add New Gesture", "Enter name for the new gesture:")

    # Update labels_dict with the new gesture label and name
    labels_dict[new_gesture_label] = new_gesture_name

    with open(labels_dict_file, 'w') as json_file:
        json.dump(labels_dict, json_file)

    # Display a message
    messagebox.showinfo("Gesture Added", f"Gesture '{new_gesture_name}' added with label '{new_gesture_label}'.\nModel calibration is in progress...")

    # Run calibrate_model() function
    calibrate_model()

    # Resume the processing loop
    pause_event.clear()

    # Display a completion message
    messagebox.showinfo("Gesture Added", "New gesture added and model calibration is done.")

# Add a global variable to store the reference to the view_window
view_window = None

# function to view gestures
def view_gestures():
    global labels_dict, view_window
    # Load labels_dict
    load_labels_dict()
    # Close the existing view window if it exists
    if view_window:
        view_window.destroy()

    # Create a new window
    view_window = tk.Toplevel(root)
    view_window.title("View Gestures")

    # Determine the number of columns based on the available space
    window_width = view_window.winfo_screenwidth()
    max_columns = window_width // 300  # Assuming each gesture image is 300 pixels wide with some padding

    # Create a canvas to dynamically adjust the layout
    canvas = tk.Canvas(view_window)
    canvas.pack()

    # List to store references to PhotoImage objects
    image_references = []

    # Function to handle gesture deletion
    def delete_gesture(gesture_index):
        global labels_dict

        # Get the name of the gesture to be deleted
        gesture_name = labels_dict.get(str(gesture_index), f"Gesture {gesture_index}")

        # Ask for confirmation
        confirmation = messagebox.askyesno("Delete Gesture", f"Are you sure you want to delete the gesture '{gesture_name}'?")

        if confirmation:
            # Adjust labels in labels_dict and JSON file
            labels_dict.pop(str(gesture_index), None)

            # Update labels_dict for remaining gestures
            for i in range(gesture_index + 1, len(labels_dict) + 1):
                labels_dict[str(i - 1)] = labels_dict.pop(str(i), f"Gesture {i}")

            with open(labels_dict_file, 'w') as json_file:
                json.dump(labels_dict, json_file)

            # Remove the corresponding data folder
            dataset_path = "./data"
            gesture_folder_path = os.path.join(dataset_path, str(gesture_index))

            # Check if the folder exists before attempting to delete
            if os.path.exists(gesture_folder_path):
                shutil.rmtree(gesture_folder_path)

            # Reorganize the data folders
            dataset_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

            # Sort the dataset folders based on their numeric labels
            dataset_folders.sort(key=lambda x: int(x))

            for i, folder in enumerate(dataset_folders):
                old_folder_path = os.path.join(dataset_path, folder)
                new_folder_path = os.path.join(dataset_path, str(i))

                # Rename the folder to adjust the labels
                os.rename(old_folder_path, new_folder_path)

            # Recalibrate the model
            calibrate_model()

            # Reopen the "View Gestures" window
            view_gestures()

    # Loop through dataset folders and display information
    dataset_path = "./data"
    dataset_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    # Sort the dataset folders based on their numeric labels
    dataset_folders.sort(key=lambda x: int(x))

    # Calculate the maximum number of rows needed
    max_rows = (len(dataset_folders) // max_columns) + 1

    # Calculate dynamic image width and height
    image_width = 200  
    image_height = 200  

    for i, dataset_folder in enumerate(dataset_folders):
        gesture_label = str(i)  # Labels start from 0
        gesture_name = labels_dict.get(gesture_label, f"Gesture {gesture_label}")

        # Load and display the first image in the dataset folder
        image_path = os.path.join(dataset_path, dataset_folder, "0.jpg")
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_width, image_height))  # Resize for display

            # Convert OpenCV image to PhotoImage format
            image_tk = cv2_to_tkinter_photoimage(image)

            # Store the reference to the PhotoImage object
            image_references.append(image_tk)

           # Calculate row and column positions
            row_position = i // max_columns
            col_position = i % max_columns * 3  # Adjusted multiplier for better spacing

            # Create a frame for each dataset
            frame = tk.Frame(canvas)
            frame.grid(row=row_position, column=col_position, padx=10, pady=10, columnspan=3)  

            # Display gesture name
            label = tk.Label(frame, text=f"{gesture_label}: {gesture_name}")
            label.grid(row=0, column=0, padx=10, pady=10, columnspan=3)  

            # Display image
            img_label = tk.Label(frame, image=image_tk)
            img_label.grid(row=1, column=0, padx=10, pady=10, columnspan=3) 

             # Add "Edit Gesture Name" button
            edit_button = ttk.Button(frame, text="Edit Gesture Name", command=lambda i=i: edit_gesture_name(i))
            edit_button.grid(row=2, column=1, pady=5, padx=(0, 10))  # Added padx for spacing

            # Add "Delete Gesture" button
            delete_button = ttk.Button(frame, text="Delete Gesture", command=lambda i=i: delete_gesture(i))
            delete_button.grid(row=2, column=2, pady=5)  

    # Keep a reference to the image objects to prevent garbage collection
    view_window.image_references = image_references

# Helper function to edit gesture name
def edit_gesture_name(gesture_index):
    global labels_dict, view_window

    # Get the current name of the gesture
    current_gesture_name = labels_dict.get(str(gesture_index), f"Gesture {gesture_index}")

    # Ask the user for the new gesture name
    new_gesture_name = simpledialog.askstring("Edit Gesture Name", f"Edit name for Gesture {gesture_index}:", initialvalue=current_gesture_name)

    # Update labels_dict with the new gesture name
    labels_dict[str(gesture_index)] = new_gesture_name

    # Update the labels_dict JSON file
    with open(labels_dict_file, 'w') as json_file:
        json.dump(labels_dict, json_file)

    # Display a message
    messagebox.showinfo("Gesture Name Edited", f"Gesture {gesture_index} name updated to '{new_gesture_name}'.")

    # Reopen the "View Gestures" window
    view_gestures()

# Helper function to convert OpenCV image to PhotoImage
def cv2_to_tkinter_photoimage(cv2_image):
    b, g, r = cv2.split(cv2_image)
    img = cv2.merge((r, g, b))
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img)
    return img_tk

# Function to toggle the visibility of the gesture frame
def toggle_gesture_buttons():
    gesture_frame.pack_forget() if gesture_frame.winfo_ismapped() else gesture_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 30))

calibrate_button = ttk.Button(root, text="Calibrate Model", command=calibrate_model)
calibrate_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 30))

reset_labels_button = ttk.Button(gesture_frame, text="Reset Gesture Names", command=reset_dataset_labels)
reset_labels_button.pack(side=tk.LEFT, padx=10)

set_timer_button = ttk.Button(root, text="Set Timer", command=set_timer)
set_timer_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 30))


start_record_button = ttk.Button(root, text="Start Recording", command=start_recording)
start_record_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 30))

stop_record_button = ttk.Button(root, text="Stop Recording", command=stop_recording)
stop_record_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 30))

tts_button = ttk.Button(root, text="Text To Speech", command=text_to_speech_realtime)
tts_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 30))

add_gesture_button = ttk.Button(gesture_frame, text="Add Gesture", command=add_gesture)
add_gesture_button.pack(side=tk.LEFT, padx=10)

view_gestures_button = ttk.Button(gesture_frame, text="View Gestures", command=view_gestures)
view_gestures_button.pack(side=tk.LEFT, padx=15)

gestures_button = ttk.Button(root, text="Gestures", command=toggle_gesture_buttons)
gestures_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(20, 30))


# Function to process video in the background
def video_processing():
    global running, results, prediction, last_gesture, last_prediction_time, labels_dict, processing_enabled, pause_event, recording, captured_inputs, predicted_textbox
    start_time_same_prediction = 0  # Variable to store the start time of the same prediction
    start_time_changed_prediction = 0  # Variable to store the start time when prediction changes
    capture_duration_changed_prediction = 1  # Duration for capturing input after the prediction changes (in seconds)

    current_gesture = None  # Initialize current_gesture outside of the recording block

    while running:
        if not processing_enabled:
            time.sleep(0.1)
            continue

        if pause_event.is_set():
            time.sleep(0.1)
            continue

        # Reload labels_dict
        load_labels_dict()

        # Load the model again
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            try:
                predicted_character = labels_dict[str(prediction[0])]
            except (NameError, KeyError):
                predicted_character = "Label Not Defined"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            # Use the prediction during recording
            if recording:
                current_gesture = predicted_character

                # Check if the prediction has remained the same for the specified duration
                if current_gesture == last_gesture:
                    if (time.time() - start_time_same_prediction) >= capture_duration_same_prediction:
                        # Capture input after the specified duration of the same prediction
                        captured_inputs.append(current_gesture)
                        predicted_textbox.delete(1.0, tk.END)  # Clear the previous content
                        predicted_textbox.insert(tk.END, ', '.join(captured_inputs))
                        start_time_same_prediction = time.time()  # Reset the start time

                # Check if the prediction has changed
                elif current_gesture != last_gesture:
                    if (time.time() - start_time_changed_prediction) >= capture_duration_changed_prediction:
                        # Capture input after the specified duration of the changed prediction
                        captured_inputs.append(current_gesture)
                        predicted_textbox.delete(1.0, tk.END)  # Clear the previous content
                        predicted_textbox.insert(tk.END, ', '.join(captured_inputs))
                        start_time_changed_prediction = time.time()  # Reset the start time

                        # Update last_gesture only if it has changed
                        last_gesture = current_gesture
                        start_time_same_prediction = time.time()  # Reset the start time for the same prediction

        cv2.imshow('Video Stream', frame)
        key = cv2.waitKey(1)

        # Break the loop if 'q' is pressed
        if key == ord('q') or cv2.getWindowProperty('Video Stream', cv2.WND_PROP_VISIBLE) < 1:
            break
        
# Create a thread for video processing
video_thread = Thread(target=video_processing)
video_thread.start()

# Start the Tkinter main loop
root.mainloop()

# Wait for the video processing thread to finish before exiting
video_thread.join()
