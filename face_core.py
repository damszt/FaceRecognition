"""
face_core.py — Core face recognition logic.

Uses storage.py for persistent file storage (Vercel Blob / local filesystem)
and database.py for attendance records (Supabase / local CSV).
"""
# pyrefly: ignore [missing-import]
import cv2
import os
import numpy as np
import base64
import json
from datetime import datetime

import storage
import database

# Absolute directory where this script lives
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_cascade_path():
    """Find the Haar cascade XML file."""
    # First try: bundled XML next to this script (most reliable on Vercel)
    local_path = os.path.join(_SCRIPT_DIR, "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return local_path
    # Second try: cwd-relative
    cwd_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
    if os.path.exists(cwd_path):
        return cwd_path
    # Last fallback: OpenCV data dir
    return cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _detect_face(image_data_base64):
    """
    Decode base64 image and detect face.
    Returns (gray_face_image, full_gray_image, faces) or (None, None, []).
    """
    try:
        encoded_data = image_data_base64.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, None, []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(get_cascade_path())
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )

        if len(faces) == 0:
            return None, gray, []

        (x, y, w, h) = faces[0]
        face_img = gray[y : y + h, x : x + w]
        return face_img, gray, faces
    except Exception as e:
        print(f"Face detection error: {e}")
        return None, None, []


def save_face_image(person_name, image_data_base64):
    """
    Decodes base64 image, detects face, and saves cropped face to storage.
    Returns True if face detected and saved, False otherwise.
    """
    try:
        face_img, _, faces = _detect_face(image_data_base64)
        if face_img is None or len(faces) == 0:
            return False

        # Encode face image to JPEG bytes
        _, buffer = cv2.imencode(".jpg", face_img)
        image_bytes = buffer.tobytes()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{timestamp}.jpg"

        # Upload to storage (Blob or local)
        result = storage.upload_dataset_image(person_name, image_bytes, filename)
        return result is not None
    except Exception as e:
        print(f"Error saving face image: {e}")
        return False


def train_model():
    """
    Downloads dataset images from storage, trains LBPH recognizer,
    and uploads model + labels back to storage.
    Returns a summary string.
    """
    try:
        # Download all dataset images from storage
        all_images = storage.download_all_dataset_images()

        if not all_images:
            return "No training data found. Please register faces first."

        faces = []
        ids = []
        label_map = {}
        current_id = 0

        for person_name, image_list in all_images.items():
            label_map[current_id] = person_name
            for filename, image_bytes in image_list:
                try:
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        ids.append(current_id)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            current_id += 1

        if not faces:
            return "No valid training images found."

        # Train recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))

        # Save to /tmp first, then upload to storage
        tmp_model = "/tmp/model.yml"
        tmp_labels = "/tmp/labels.npy"

        recognizer.save(tmp_model)
        np.save(tmp_labels, label_map)

        # Read bytes and upload
        with open(tmp_model, "rb") as f:
            model_bytes = f.read()
        with open(tmp_labels, "rb") as f:
            labels_bytes = f.read()

        model_url, labels_url = storage.upload_model(model_bytes, labels_bytes)

        if model_url and labels_url:
            return f"Training complete. Trained on {len(faces)} images for {len(label_map)} people."
        else:
            return f"Training complete locally but failed to upload to storage. Trained on {len(faces)} images."
    except Exception as e:
        return f"Training failed: {e}"


# Global recognizer and labels to avoid reloading every request
_recognizer = None
_labels_map = None


def load_resources():
    """
    Load model and labels from storage into memory.
    Downloads from Blob Storage to /tmp if needed.
    Returns True on success.
    """
    global _recognizer, _labels_map

    model_path, labels_path = storage.download_model()

    if model_path and labels_path:
        try:
            _recognizer = cv2.face.LBPHFaceRecognizer_create()
            _recognizer.read(model_path)
            _labels_map = np.load(labels_path, allow_pickle=True).item()
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False
    return False


def recognize_face(image_data_base64):
    """
    Recognizes face from base64 image.
    Returns (name, confidence, details) or (None, None, None).
    """
    global _recognizer, _labels_map

    if _recognizer is None or _labels_map is None:
        if not load_resources():
            return None, None, None

    try:
        face_img, _, faces = _detect_face(image_data_base64)
        if face_img is None or len(faces) == 0:
            return None, None, None

        # Use StandardCollector to get all results
        collector = cv2.face.StandardCollector_create()
        _recognizer.predict_collect(face_img, collector)
        results = collector.getResults(sorted=True)

        details = []
        for label, dist in results:
            name = _labels_map.get(label, "Unknown")
            details.append({"name": name, "distance": dist})

        best_label = results[0][0]
        best_dist = results[0][1]
        best_name = _labels_map.get(best_label, "Unknown")

        return best_name, best_dist, details
    except Exception as e:
        print(f"Recognition error: {e}")
        return None, None, None


def log_attendance(name, confidence, details=None):
    """Log attendance to Supabase (or local CSV fallback)."""
    return database.log_attendance(name, confidence, details)


def get_attendance_logs(date_str=None):
    """Get attendance logs from Supabase (or local CSV fallback)."""
    return database.get_attendance_logs(date_str)


def get_model_stats():
    """Returns statistics about the dataset and model."""
    people = storage.list_people()
    total_images, image_counts = storage.count_images()

    stats = {
        "total_people": len(people),
        "total_images": total_images,
        "last_trained": "Never",
        "people_names": people,
    }

    # Check if model exists and get its timestamp
    if storage.model_exists():
        # For local dev, check file mtime
        if not storage.IS_VERCEL:
            if os.path.exists(storage.LOCAL_MODEL_FILE):
                mtime = os.path.getmtime(storage.LOCAL_MODEL_FILE)
                stats["last_trained"] = datetime.fromtimestamp(mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                stats["last_trained"] = "Model in cloud"
        else:
            stats["last_trained"] = "Model in cloud"

    return stats


def clear_dataset(person_name=None):
    """
    Clears registered datasets from storage.
    If person_name is provided, only that person is deleted.
    Otherwise, all datasets and model files are deleted.
    Returns (True, message) on success, (False, error_message) otherwise.
    """
    global _recognizer, _labels_map

    try:
        if person_name:
            success = storage.delete_person_dataset(person_name)
            if success:
                return True, f"Dataset for '{person_name}' has been deleted."
            else:
                return False, f"Could not find dataset for '{person_name}'."
        else:
            ds_ok = storage.delete_all_datasets()
            model_ok = storage.delete_model()

            _recognizer = None
            _labels_map = None

            if ds_ok and model_ok:
                return True, "All datasets and trained models have been cleared."
            elif ds_ok:
                return True, "Datasets cleared, but model deletion had issues."
            else:
                return False, "Error clearing datasets."
    except Exception as e:
        error_msg = f"Error clearing dataset: {e}"
        print(error_msg)
        return False, error_msg
