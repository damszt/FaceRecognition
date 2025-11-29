import cv2
import os
import numpy as np
import base64
import csv
from datetime import datetime

# Constants
DATASET_DIR = "dataset"
MODEL_FILE = "model.yml"
LABELS_FILE = "labels.npy"
ATTENDANCE_FILE_PREFIX = "attendance_"

# Ensure dataset directory exists
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def save_face_image(person_name, image_data_base64):
    """
    Decodes base64 image, detects face, and saves it to dataset/{person_name}/.
    Returns True if face detected and saved, False otherwise.
    """
    try:
        # Decode base64 image
        encoded_data = image_data_base64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return False

        # Create person directory if not exists
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        # Save the first detected face
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{person_dir}/img_{timestamp}.jpg"
        cv2.imwrite(filename, face_img)
        
        return True
    except Exception as e:
        print(f"Error saving face image: {e}")
        return False

def train_model():
    """
    Trains the LBPH recognizer using images in dataset/.
    Saves model.yml and labels.npy.
    Returns a summary string.
    """
    try:
        if not os.path.exists(DATASET_DIR):
            return "Dataset directory not found."

        faces = []
        ids = []
        label_map = {}
        current_id = 0
        
        # Traverse dataset directory
        for person_name in os.listdir(DATASET_DIR):
            person_path = os.path.join(DATASET_DIR, person_name)
            if not os.path.isdir(person_path):
                continue
                
            label_map[current_id] = person_name
            
            for image_name in os.listdir(person_path):
                if image_name.startswith("."): continue # Skip hidden files
                
                image_path = os.path.join(person_path, image_name)
                try:
                    # Read image in grayscale
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    
                    faces.append(img)
                    ids.append(current_id)
                except Exception as e:
                    print(f"Error reading {image_path}: {e}")
            
            current_id += 1

        if not faces:
            return "No training data found."

        # Train recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        
        # Save model and labels
        recognizer.save(MODEL_FILE)
        np.save(LABELS_FILE, label_map)
        
        return f"Training complete. Trained on {len(faces)} images for {len(label_map)} people."
    except Exception as e:
        return f"Training failed: {e}"

# Global recognizer and labels to avoid reloading every request
recognizer = None
labels_map = None

def load_resources():
    global recognizer, labels_map
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(MODEL_FILE)
            labels_map = np.load(LABELS_FILE, allow_pickle=True).item()
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False
    return False

import json

# ... (imports remain the same)

# ... (previous code)

def recognize_face(image_data_base64):
    """
    Recognizes face from base64 image.
    Returns (name, confidence, details) or (None, None, None).
    """
    global recognizer, labels_map
    
    if recognizer is None or labels_map is None:
        if not load_resources():
            return None, None, None

    try:
        encoded_data = image_data_base64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None, None, None

        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        # Use StandardCollector to get all results
        collector = cv2.face.StandardCollector_create()
        recognizer.predict_collect(face_img, collector)
        results = collector.getResults(sorted=True)
        
        details = []
        for label, dist in results:
            name = labels_map.get(label, "Unknown")
            details.append({"name": name, "distance": dist})
            
        # Best match is the first one
        best_label = results[0][0]
        best_dist = results[0][1]
        
        best_name = labels_map.get(best_label, "Unknown")
            
        return best_name, best_dist, details

    except Exception as e:
        print(f"Recognition error: {e}")
        return None, None, None

def log_attendance(name, confidence, details=None):
    """
    Logs attendance to CSV if not already logged for today.
    Returns True if logged, False if already present or error.
    """
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{ATTENDANCE_FILE_PREFIX}{today_str}.csv"
        
        # Check if already logged
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == name:
                        return False # Already logged
        
        # Log attendance
        timestamp = datetime.now().strftime("%H:%M:%S")
        confidence_str = f"{confidence:.2f}"
        details_json = json.dumps(details) if details else "[]"
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp, confidence_str, details_json])
            
        return True
    except Exception as e:
        print(f"Logging error: {e}")
        return False

def get_attendance_logs(date_str=None):
    """
    Returns list of attendance records for a given date (YYYY-MM-DD).
    If date_str is None, returns today's logs.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    filename = f"{ATTENDANCE_FILE_PREFIX}{date_str}.csv"
    logs = []
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        confidence = row[2] if len(row) > 2 else "N/A"
                        details = json.loads(row[3]) if len(row) > 3 else []
                        logs.append({
                            "name": row[0], 
                            "timestamp": row[1], 
                            "confidence": confidence,
                            "details": details
                        })
        except Exception as e:
            print(f"Error reading logs: {e}")
            
    return logs

def get_model_stats():
    """
    Returns statistics about the dataset and model.
    """
    stats = {
        "total_people": 0,
        "total_images": 0,
        "last_trained": "Never"
    }
    
    if os.path.exists(DATASET_DIR):
        people = [p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))]
        stats["total_people"] = len(people)
        
        count = 0
        for p in people:
            path = os.path.join(DATASET_DIR, p)
            count += len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        stats["total_images"] = count
        
    if os.path.exists(MODEL_FILE):
        mtime = os.path.getmtime(MODEL_FILE)
        stats["last_trained"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        
    return stats
