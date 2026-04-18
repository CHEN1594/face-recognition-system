import os
import cv2

DATASET_DIR = "dataset/train"
CAMERA_INDEX = 0
TARGET_SIZE = 20  # number of images to capture

def main():
    person_name = input("Enter person name: ").strip()
    if not person_name:
        print("Name cannot be empty.")
        return

    save_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press SPACE to capture an image.")
    print("Press Q to quit.")

    count = len([f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        display = frame.copy()
        cv2.putText(display, f"Person: {person_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {count}/{TARGET_SIZE}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Faces", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            filename = os.path.join(save_dir, f"{count+1:03d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"Saved {filename}")
            if count >= TARGET_SIZE:
                print("Target reached.")
                break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()