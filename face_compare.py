import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle
from sklearn.metrics.pairwise import cosine_similarity

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

def capture_faces(image_data):
    """Capture faces and save embeddings from image data"""
    embeddings_dict = load_embeddings()

    # Decode the image data from base64
    import base64
    import numpy as np
    import cv2

    # Convert the image data to a numpy array
    image_data = image_data.split(",")[1]  # Remove the metadata
    image_data = base64.b64decode(image_data)
    np_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        print("Error: Decoded frame is empty.")
        return {'error': 'Invalid image data'}

    faces = arcface.get(frame)

    if len(faces) > 0:
        face_embedding = faces[0].normed_embedding
        
        while True:
            label = input("Enter unique label for this face: ").strip()
            if not label:
                print("Label cannot be empty. Please try again.")
                continue
            if label not in embeddings_dict:
                break
            print(f"Label '{label}' already exists. Please enter a unique label.")
        
        embeddings_dict[label] = face_embedding
        if save_embeddings(embeddings_dict):
            print(f"Face '{label}' captured successfully.")
        else:
            print("Failed to save embedding. Please try again.")
    else:
        print("No face detected. Please try again.")

def compare_faces(image_data):
    print(f"Received image data length: {len(image_data)}")  # Debugging statement
    print(f"Image data (first 30 chars): {image_data[:30]}")  # Debugging statement
    """Compare captured faces with saved embeddings from image data"""
    embeddings_dict = load_embeddings()

    # Decode the image data from base64
    import base64
    import numpy as np
    import cv2

    # Convert the image data to a numpy array
    image_data = image_data.split(",")[1]  # Remove the metadata
    image_data = base64.b64decode(image_data)
    np_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    faces = arcface.get(frame)

    for face in faces:
        face_embedding = face.normed_embedding
        face_box = face.bbox.astype(int)
        match_found = False
        best_match = {"label": "Unknown", "similarity": 0}
        
        for label, saved_embedding in embeddings_dict.items():
            similarity = cosine_similarity([face_embedding], [saved_embedding])[0][0]
            if similarity > 0.6 and similarity > best_match["similarity"]:
                best_match = {"label": label, "similarity": similarity}
                match_found = True
        
        color = (0, 255, 0) if match_found else (0, 0, 255)
        label = f"{best_match['label']} ({best_match['similarity']:.2f})" if match_found else "No Match"
        cv2.rectangle(frame, (face_box[0], face_box[1]), 
                      (face_box[2], face_box[3]), color, 2)
        cv2.putText(frame, label, (face_box[0], face_box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Convert the frame to base64 for response if frame is valid
    _, buffer = cv2.imencode('.png', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'image': f"data:image/png;base64,{frame_base64}",
        'matches': best_match
    }

if __name__ == "__main__":
    capture_faces()  # or call compare_faces() as needed
