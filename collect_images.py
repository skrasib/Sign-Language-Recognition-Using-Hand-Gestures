import os
import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1  # You may adjust this if collecting data for multiple classes
dataset_size = 100

# Determine the next available index dynamically
existing_classes = [int(d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
next_class_index = max(existing_classes, default=-1) + 1

# Find the highest existing class index
highest_existing_index = max(existing_classes, default=-1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Use the next available index after the highest existing index
next_class_index = highest_existing_index + 1

for j in range(next_class_index, next_class_index + number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for 'Q' key press to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start capturing!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        image_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(image_path, frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
