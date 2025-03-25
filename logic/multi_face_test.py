import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Initialize Face Recognition Model
arcface = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
arcface.prepare(ctx_id=0)

# ✅ Load stored embeddings
def load_embeddings():
    try:
        with open("face_embeddings.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

embeddings_dict = load_embeddings()

# ✅ Function to process frame
def process_face(frame):
    print("Processing frame...")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = arcface.get(frame_rgb)

    print(f"Faces detected: {len(faces)}")  # Debug log

    results = []

    for face in faces:
        face_embedding = face.normed_embedding
        face_box = face.bbox.astype(int)

        # Compare with stored embeddings
        best_match = {"label": "Unknown", "similarity": 0}
        for label, saved_embedding in embeddings_dict.items():
            similarity = cosine_similarity([face_embedding], [saved_embedding])[0][0]
            if similarity > 0.6 and similarity > best_match["similarity"]:
                best_match = {"label": label, "similarity": similarity}

        print(f"Match Found: {best_match['label']} ({best_match['similarity']})")

        results.append({
            "label": best_match["label"],
            "similarity": best_match["similarity"],
            "bbox": face_box.tolist()
        })

    if not results:  # If no faces are found, return an empty list
        return {"faces": []}

    return {"faces": results}
