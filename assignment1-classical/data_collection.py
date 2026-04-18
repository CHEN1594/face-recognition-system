import cv2
import os

def capture_face_data(person_name, num_images=15):
    dataset_dir = f"gallery/{person_name}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)
    
    print(f"--- LIVE CAMERA STARTED ---")
    print(f"1. Window might open behind VS Code. Check your taskbar!")
    print(f"2. Press the SPACEBAR to capture an image ({num_images} needed).")
    print(f"3. Press 'q' to quit.")
    
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        # Draw a rectangle around detected faces in the live feed
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add a counter to the screen
            cv2.putText(frame, f"Captured: {count}/{num_images}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the live camera feed
        cv2.imshow('Face Data Collection - PRESS SPACE TO CAPTURE', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If Spacebar (key 32) is pressed, AND a face is detected, save the image
        if key == 32:
            if len(faces) > 0:
                # Take the first face detected
                (x, y, w, h) = faces[0]
                cropped_face = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(cropped_face, (90, 90))
                
                file_path = f"{dataset_dir}/{person_name}_{count}.jpg"
                cv2.imwrite(file_path, resized_face)
                count += 1
                print(f" SNAP! Captured {count}/{num_images} for {person_name}")
            else:
                print(" No face detected! Make sure you are in the frame.")
                
        # If 'q' is pressed, quit early
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete for {person_name}!")

if __name__ == "__main__":
    member_name = input("Enter the team member's name: ")
    capture_face_data(member_name, num_images=15)
