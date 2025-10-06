import os
import cv2
import subprocess
import sys
from tqdm import tqdm  # Import tqdm

def run_module_command(module_name):
    """
    Executes a Python module and streams its output in real-time
    to show live logs.
    """
    print(f"\n--- Running module: {module_name} ---")
    command = [sys.executable, "-m", module_name]
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )

        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print(f"\nError: Module {module_name} exited with return code {return_code}")
        
        print(f"--- Finished module: {module_name} ---")

    except Exception as e:
        print(f"An unexpected error occurred while trying to run {module_name}: {e}")


def create_classes_and_capture_images():
    """
    Prompts the user to create classes (directories) and captures images for each class.
    """
    base_dir = "dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    while True:
        class_name = input("Enter class name (e.g., 'person_A') or type 'done' to finish: ")
        if class_name.lower() == 'done':
            break

        class_path = os.path.join(base_dir, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            print(f"Directory '{class_path}' created.")

        print(f"\nStarting image capture for class: {class_name}")
        capture_images(class_path)

    print("\nImage capture session finished.")


def capture_images(class_path):
    """
    Opens a CV2 window and captures images for a given class.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    img_counter = 0
    min_images = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        text = f"Class: {os.path.basename(class_path)} | Press 'C' to capture ({img_counter}/{min_images})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'Q' to quit this class.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Image Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_name = os.path.join(class_path, f"{img_counter}.png")
            cv2.imwrite(img_name, frame)
            print(f"Image {img_name} saved!")
            img_counter += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if img_counter < min_images:
        print(f"\nWarning: Captured {img_counter} images. Less than the minimum of {min_images}.\n")
    else:
        print(f"\nSuccessfully captured {img_counter} images for '{os.path.basename(class_path)}'.\n")


if __name__ == "__main__":
    # Step 1: Guide user to create classes and capture images
    create_classes_and_capture_images()

    # Step 2: Define the sequence of modules to run
    modules_to_run = [
        "src.data_preprocessing",
        "src.augment",
        "src.train",
        "src.evaluate"
    ]

    # Use tqdm to create a progress bar for the pipeline
    print("\nStarting the processing pipeline...")
    for module in tqdm(modules_to_run, desc="Pipeline Progress"):
        run_module_command(module)

    # Step 3: Print final instructions for the user
    print("\n\n--- Pipeline Complete ---")
    print("To test the model, type the following command in your terminal (from the parent directory):")
    print("\npython -m src.recognise image_name.jpg\n")
    print("The file should be in the same directory as this file.")