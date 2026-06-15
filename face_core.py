# pyrefly: ignore [missing-import]
import cv2
import os
import numpy as np
import base64
import csv
from datetime import datetime

# Constants
IS_VERCEL = "VERCEL" in os.environ or "AWS_LAMBDA_FUNCTION_NAME" in os.environ

if IS_VERCEL:
    DATASET_DIR = "/tmp/dataset"
    MODEL_FILE = "/tmp/model.yml"
    LABELS_FILE = "/tmp/labels.npy"
    ATTENDANCE_FILE_PREFIX = "/tmp/attendance_"
else:
    DATASET_DIR = "dataset"
    MODEL_FILE = "model.yml"
    LABELS_FILE = "labels.npy"
    ATTENDANCE_FILE_PREFIX = "attendance_"

def get_model_path():
    # Prefer root/script-dir file (deployed via git) over /tmp (ephemeral)
    root_model = os.path.join(_SCRIPT_DIR, "model.yml")
    if os.path.exists(root_model):
        return root_model
    # Fall back to /tmp version (after training on Vercel)
    if os.path.exists(MODEL_FILE):
        return MODEL_FILE
    return MODEL_FILE

def get_labels_path():
    # Prefer root/script-dir file (deployed via git) over /tmp (ephemeral)
    root_labels = os.path.join(_SCRIPT_DIR, "labels.npy")
    if os.path.exists(root_labels):
        return root_labels
    # Fall back to /tmp version (after training on Vercel)
    if os.path.exists(LABELS_FILE):
        return LABELS_FILE
    return LABELS_FILE

# Absolute directory where this script lives — works on Vercel too
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_cascade_path():
    # First try: bundled XML next to this script (most reliable on Vercel)
    local_path = os.path.join(_SCRIPT_DIR, "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return local_path
    # Second try: cwd-relative
    cwd_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
    if os.path.exists(cwd_path):
        return cwd_path
    # Last fallback: OpenCV data dir
    return cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

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
        face_cascade = cv2.CascadeClassifier(get_cascade_path())
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

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
        # Determine dataset directories to check
        dirs_to_check = []
        if IS_VERCEL:
            if os.path.exists("/tmp/dataset"):
                dirs_to_check.append("/tmp/dataset")
            if os.path.exists("dataset"):
                dirs_to_check.append("dataset")
        else:
            if os.path.exists(DATASET_DIR):
                dirs_to_check.append(DATASET_DIR)
                
        if not dirs_to_check:
            return "Dataset directory not found."

        faces = []
        ids = []
        label_map = {}
        current_id = 0
        
        # Traverse all checked dataset directories, grouping by person name
        people_images = {}
        for d in dirs_to_check:
            for person_name in os.listdir(d):
                person_path = os.path.join(d, person_name)
                if not os.path.isdir(person_path):
                    continue
                if person_name not in people_images:
                    people_images[person_name] = []
                for image_name in os.listdir(person_path):
                    if image_name.startswith("."):
                        continue
                    people_images[person_name].append(os.path.join(person_path, image_name))
        
        if not people_images:
            return "No training data found."

        for person_name, image_paths in people_images.items():
            label_map[current_id] = person_name
            for image_path in image_paths:
                try:
                    # Read image in grayscale
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
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
    model_path = get_model_path()
    labels_path = get_labels_path()
    if os.path.exists(model_path) and os.path.exists(labels_path):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(model_path)
            labels_map = np.load(labels_path, allow_pickle=True).item()
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False
    return False

import json



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
        face_cascade = cv2.CascadeClassifier(get_cascade_path())
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

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
        
        # On Vercel, if today's attendance log is in the root directory but not yet in /tmp,
        # copy it to /tmp so we can read and append to it.
        if IS_VERCEL and not os.path.exists(filename):
            root_filename = f"attendance_{today_str}.csv"
            if os.path.exists(root_filename):
                import shutil
                try:
                    shutil.copy(root_filename, filename)
                except Exception as e:
                    print(f"Error copying attendance log: {e}")

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
    
    # On Vercel, if the log file does not exist in /tmp, check if it was pre-packaged in root
    if IS_VERCEL and not os.path.exists(filename):
        root_filename = f"attendance_{date_str}.csv"
        if os.path.exists(root_filename):
            filename = root_filename

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
    Falls back to reading registered people from labels.npy if dataset folder is empty.
    """
    stats = {
        "total_people": 0,
        "total_images": 0,
        "last_trained": "Never"
    }

    # Check dataset directories
    dirs_to_check = []
    if IS_VERCEL:
        if os.path.exists("/tmp/dataset"):
            dirs_to_check.append("/tmp/dataset")
        root_dataset = os.path.join(_SCRIPT_DIR, "dataset")
        if os.path.exists(root_dataset):
            dirs_to_check.append(root_dataset)
    else:
        if os.path.exists(DATASET_DIR):
            dirs_to_check.append(DATASET_DIR)

    people_sets = set()
    total_images = 0

    for d in dirs_to_check:
        people = [p for p in os.listdir(d) if os.path.isdir(os.path.join(d, p))]
        people_sets.update(people)
        for p in people:
            path = os.path.join(d, p)
            total_images += len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    stats["total_people"] = len(people_sets)
    stats["total_images"] = total_images

    # If dataset folder is empty but model exists, read people count from labels.npy
    if stats["total_people"] == 0:
        labels_path = get_labels_path()
        if os.path.exists(labels_path):
            try:
                label_map = np.load(labels_path, allow_pickle=True).item()
                stats["total_people"] = len(label_map)
                stats["total_images"] = -1  # Unknown (model deployed without dataset)
                stats["people_names"] = list(label_map.values())
            except Exception as e:
                print(f"Could not read labels: {e}")

    model_path = get_model_path()
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        stats["last_trained"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")


    return stats

def clear_dataset(person_name=None):
    """
    Clears registered datasets.
    If person_name is provided, deletes the specific person's dataset folder inside DATASET_DIR.
    If person_name is None, deletes all person folders inside DATASET_DIR.
    Also deletes the trained model files (MODEL_FILE and LABELS_FILE) if they exist.
    On Vercel, also clears root-level dataset/ and model files since /tmp is ephemeral.
    Returns (True, message) on success, (False, error_message) otherwise.
    """
    try:
        import shutil
        deleted_count = 0

        # Determine all dataset directories to clear
        dataset_dirs_to_clear = [DATASET_DIR]
        if IS_VERCEL:
            # Also clear root-level dataset dir in case it's the active one
            root_dataset = os.path.join(_SCRIPT_DIR, "dataset")
            if root_dataset != DATASET_DIR and os.path.exists(root_dataset):
                dataset_dirs_to_clear.append(root_dataset)

        if person_name:
            for d in dataset_dirs_to_clear:
                person_dir = os.path.join(d, person_name)
                if os.path.exists(person_dir):
                    shutil.rmtree(person_dir)
                    deleted_count += 1
        else:
            for d in dataset_dirs_to_clear:
                if os.path.exists(d):
                    for entry in os.listdir(d):
                        path = os.path.join(d, entry)
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                            deleted_count += 1

            # Delete model files — both /tmp and root locations
            model_files_to_delete = [MODEL_FILE]
            labels_files_to_delete = [LABELS_FILE]
            if IS_VERCEL:
                root_model = os.path.join(_SCRIPT_DIR, "model.yml")
                root_labels = os.path.join(_SCRIPT_DIR, "labels.npy")
                if root_model not in model_files_to_delete:
                    model_files_to_delete.append(root_model)
                if root_labels not in labels_files_to_delete:
                    labels_files_to_delete.append(root_labels)

            for mf in model_files_to_delete:
                if os.path.exists(mf):
                    try:
                        os.remove(mf)
                    except Exception as e:
                        print(f"Could not delete {mf}: {e}")
            for lf in labels_files_to_delete:
                if os.path.exists(lf):
                    try:
                        os.remove(lf)
                    except Exception as e:
                        print(f"Could not delete {lf}: {e}")

            global recognizer, labels_map
            recognizer = None
            labels_map = None

        msg = (f"Dataset for '{person_name}' has been deleted."
               if person_name
               else f"All datasets and trained models have been cleared. Deleted {deleted_count} person folders.")
        return True, msg
    except Exception as e:
        error_msg = f"Error clearing dataset: {e}"
        print(error_msg)
        return False, error_msg


