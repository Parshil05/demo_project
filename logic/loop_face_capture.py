import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle

# Initialize ArcFace
arcface = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
arcface.prepare(ctx_id=0)

def load_embeddings():
    """Load existing embeddings from file"""
    try:
        if os.path.exists("face_embeddings.pkl"):
            with open("face_embeddings.pkl", "rb") as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}

def save_embeddings(embeddings_dict):
    """Save embeddings to file"""
    try:
        with open("face_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_dict, f)
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    
    # Load existing embeddings
    embeddings_dict = load_embeddings()

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Detect faces in frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = arcface.get(frame_rgb)
        
        # Show frame
        cv2.imshow('Face Capture', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # Capture on 's' key press
            if len(faces) > 0:
                # Get embedding from first detected face
                face_embedding = faces[0].normed_embedding
                
                # Prompt user for unique label
                while True:
                    label = input("Enter unique label for this face: ").strip()
                    if not label:
                        print("Label cannot be empty. Please try again.")
                        continue
                    if label not in embeddings_dict:
                        break
                    print(f"Label '{label}' already exists. Please enter a unique label.")
                
                # Save the new embedding
                embeddings_dict[label] = face_embedding
                if save_embeddings(embeddings_dict):
                    print(f"Face '{label}' captured successfully.")
                else:
                    print("Failed to save embedding. Please try again.")
                
                # Ask if user wants to add another face
                while True:
                    choice = input("Do you want to add another face? (yes/no): ").lower()
                    if choice in ['yes', 'no']:
                        break
                    print("Please enter 'yes' or 'no'")
                
                if choice == 'no':
                    break
            else:
                print("No face detected. Please try again.")
        
        if key == ord('q'):  # Exit on 'q' key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face capture session ended.")

if __name__ == "__main__":
    main()
