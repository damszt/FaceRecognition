"""
storage.py — Vercel Blob Storage abstraction for Face Recognition app.

Handles upload/download of dataset images and model files.
Uses the Vercel Blob REST API directly via requests (no SDK dependency needed).
Falls back to local filesystem when not on Vercel (for local development).
"""
import os
import requests
import json

# Vercel Blob config
BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
BLOB_API_URL = "https://blob.vercel-storage.com"

# Detect environment
IS_VERCEL = "VERCEL" in os.environ or "AWS_LAMBDA_FUNCTION_NAME" in os.environ

# Local fallback directories
LOCAL_DATASET_DIR = "dataset"
LOCAL_MODEL_FILE = "model.yml"
LOCAL_LABELS_FILE = "labels.npy"


def _blob_headers():
    """Common headers for Vercel Blob API requests."""
    return {
        "Authorization": f"Bearer {BLOB_TOKEN}",
    }


# ============================================================
# Dataset Image Operations
# ============================================================

def upload_dataset_image(person_name, image_bytes, filename):
    """
    Upload a dataset image to Vercel Blob Storage.
    Returns the blob URL on success, None on failure.
    """
    if not IS_VERCEL or not BLOB_TOKEN:
        # Local fallback: save to filesystem
        person_dir = os.path.join(LOCAL_DATASET_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        filepath = os.path.join(person_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return filepath

    try:
        blob_path = f"dataset/{person_name}/{filename}"
        resp = requests.put(
            f"{BLOB_API_URL}/{blob_path}",
            headers={
                **_blob_headers(),
                "x-api-version": "7",
                "Content-Type": "image/jpeg",
            },
            data=image_bytes,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("url", blob_path)
        else:
            print(f"Blob upload error: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        print(f"Blob upload exception: {e}")
        return None


def list_people():
    """
    List all registered people (person names) from Blob Storage.
    Returns a list of person name strings.
    """
    if not IS_VERCEL or not BLOB_TOKEN:
        # Local fallback
        if os.path.exists(LOCAL_DATASET_DIR):
            return [
                d for d in os.listdir(LOCAL_DATASET_DIR)
                if os.path.isdir(os.path.join(LOCAL_DATASET_DIR, d))
            ]
        return []

    try:
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": "dataset/", "mode": "folded"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Extract unique person names from blob paths
            people = set()
            for blob_info in data.get("blobs", []):
                path = blob_info.get("pathname", "")
                parts = path.split("/")
                if len(parts) >= 2 and parts[0] == "dataset":
                    people.add(parts[1])
            # Also check folded prefixes
            for folder in data.get("folders", []):
                # folder looks like "dataset/person_name/"
                parts = folder.strip("/").split("/")
                if len(parts) >= 2:
                    people.add(parts[1])
            return list(people)
        return []
    except Exception as e:
        print(f"Blob list_people error: {e}")
        return []


def count_images(person_name=None):
    """
    Count dataset images, optionally for a specific person.
    Returns (total_images, {person: count}) dict.
    """
    if not IS_VERCEL or not BLOB_TOKEN:
        # Local fallback
        counts = {}
        if os.path.exists(LOCAL_DATASET_DIR):
            for p in os.listdir(LOCAL_DATASET_DIR):
                p_dir = os.path.join(LOCAL_DATASET_DIR, p)
                if os.path.isdir(p_dir):
                    if person_name and p != person_name:
                        continue
                    img_count = len([
                        f for f in os.listdir(p_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])
                    counts[p] = img_count
        total = sum(counts.values())
        return total, counts

    try:
        prefix = f"dataset/{person_name}/" if person_name else "dataset/"
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": prefix, "limit": 1000},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            counts = {}
            for blob_info in data.get("blobs", []):
                path = blob_info.get("pathname", "")
                parts = path.split("/")
                if len(parts) >= 3 and parts[0] == "dataset":
                    name = parts[1]
                    counts[name] = counts.get(name, 0) + 1
            total = sum(counts.values())
            return total, counts
        return 0, {}
    except Exception as e:
        print(f"Blob count_images error: {e}")
        return 0, {}


def download_all_dataset_images():
    """
    Download ALL dataset images from Blob Storage.
    Returns dict: {person_name: [(filename, image_bytes), ...]}
    """
    if not IS_VERCEL or not BLOB_TOKEN:
        # Local fallback: read from filesystem
        result = {}
        if os.path.exists(LOCAL_DATASET_DIR):
            for person_name in os.listdir(LOCAL_DATASET_DIR):
                person_dir = os.path.join(LOCAL_DATASET_DIR, person_name)
                if not os.path.isdir(person_dir):
                    continue
                result[person_name] = []
                for fname in os.listdir(person_dir):
                    if fname.startswith("."):
                        continue
                    fpath = os.path.join(person_dir, fname)
                    with open(fpath, "rb") as f:
                        result[person_name].append((fname, f.read()))
        return result

    try:
        # List all blobs under dataset/
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": "dataset/", "limit": 1000},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"Blob list error: {resp.status_code}")
            return {}

        data = resp.json()
        result = {}

        for blob_info in data.get("blobs", []):
            path = blob_info.get("pathname", "")
            url = blob_info.get("url", "")
            parts = path.split("/")
            if len(parts) < 3 or parts[0] != "dataset":
                continue

            person_name = parts[1]
            filename = parts[2]

            if person_name not in result:
                result[person_name] = []

            # Download the image bytes
            try:
                img_resp = requests.get(url, timeout=30)
                if img_resp.status_code == 200:
                    result[person_name].append((filename, img_resp.content))
            except Exception as e:
                print(f"Error downloading {url}: {e}")

        return result
    except Exception as e:
        print(f"Blob download_all error: {e}")
        return {}


def delete_person_dataset(person_name):
    """Delete all images for a specific person."""
    if not IS_VERCEL or not BLOB_TOKEN:
        import shutil
        person_dir = os.path.join(LOCAL_DATASET_DIR, person_name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            return True
        return False

    try:
        # List blobs for this person
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": f"dataset/{person_name}/", "limit": 1000},
            timeout=15,
        )
        if resp.status_code != 200:
            return False

        urls = [b.get("url") for b in resp.json().get("blobs", []) if b.get("url")]
        if not urls:
            return False

        # Delete blobs
        del_resp = requests.post(
            f"{BLOB_API_URL}/delete",
            headers={**_blob_headers(), "Content-Type": "application/json"},
            json={"urls": urls},
            timeout=30,
        )
        return del_resp.status_code == 200
    except Exception as e:
        print(f"Blob delete_person error: {e}")
        return False


def delete_all_datasets():
    """Delete all dataset images."""
    if not IS_VERCEL or not BLOB_TOKEN:
        import shutil
        if os.path.exists(LOCAL_DATASET_DIR):
            for name in os.listdir(LOCAL_DATASET_DIR):
                path = os.path.join(LOCAL_DATASET_DIR, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
        return True

    try:
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": "dataset/", "limit": 1000},
            timeout=15,
        )
        if resp.status_code != 200:
            return False

        urls = [b.get("url") for b in resp.json().get("blobs", []) if b.get("url")]
        if not urls:
            return True  # Nothing to delete

        del_resp = requests.post(
            f"{BLOB_API_URL}/delete",
            headers={**_blob_headers(), "Content-Type": "application/json"},
            json={"urls": urls},
            timeout=30,
        )
        return del_resp.status_code == 200
    except Exception as e:
        print(f"Blob delete_all error: {e}")
        return False


# ============================================================
# Model File Operations
# ============================================================

def upload_model(model_bytes, labels_bytes):
    """
    Upload trained model and labels to Vercel Blob Storage.
    Returns (model_url, labels_url) or (None, None) on failure.
    """
    if not IS_VERCEL or not BLOB_TOKEN:
        with open(LOCAL_MODEL_FILE, "wb") as f:
            f.write(model_bytes)
        with open(LOCAL_LABELS_FILE, "wb") as f:
            f.write(labels_bytes)
        return LOCAL_MODEL_FILE, LOCAL_LABELS_FILE

    try:
        model_url = None
        labels_url = None

        # Upload model.yml
        resp = requests.put(
            f"{BLOB_API_URL}/model/model.yml",
            headers={
                **_blob_headers(),
                "x-api-version": "7",
                "Content-Type": "application/octet-stream",
            },
            data=model_bytes,
            timeout=60,
        )
        if resp.status_code == 200:
            model_url = resp.json().get("url")

        # Upload labels.npy
        resp = requests.put(
            f"{BLOB_API_URL}/model/labels.npy",
            headers={
                **_blob_headers(),
                "x-api-version": "7",
                "Content-Type": "application/octet-stream",
            },
            data=labels_bytes,
            timeout=30,
        )
        if resp.status_code == 200:
            labels_url = resp.json().get("url")

        return model_url, labels_url
    except Exception as e:
        print(f"Blob upload_model error: {e}")
        return None, None


def download_model():
    """
    Download model and labels from Vercel Blob Storage to /tmp.
    Returns (model_path, labels_path) or (None, None) if not found.
    """
    if not IS_VERCEL or not BLOB_TOKEN:
        model_path = LOCAL_MODEL_FILE
        labels_path = LOCAL_LABELS_FILE
        if os.path.exists(model_path) and os.path.exists(labels_path):
            return model_path, labels_path
        return None, None

    try:
        # Check for cached files in /tmp first
        tmp_model = "/tmp/model.yml"
        tmp_labels = "/tmp/labels.npy"

        # List model blobs
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": "model/", "limit": 10},
            timeout=15,
        )
        if resp.status_code != 200:
            return None, None

        blobs = {b.get("pathname"): b.get("url") for b in resp.json().get("blobs", [])}

        model_blob_url = blobs.get("model/model.yml")
        labels_blob_url = blobs.get("model/labels.npy")

        if not model_blob_url or not labels_blob_url:
            return None, None

        # Download model
        model_resp = requests.get(model_blob_url, timeout=60)
        if model_resp.status_code == 200:
            with open(tmp_model, "wb") as f:
                f.write(model_resp.content)
        else:
            return None, None

        # Download labels
        labels_resp = requests.get(labels_blob_url, timeout=30)
        if labels_resp.status_code == 200:
            with open(tmp_labels, "wb") as f:
                f.write(labels_resp.content)
        else:
            return None, None

        return tmp_model, tmp_labels
    except Exception as e:
        print(f"Blob download_model error: {e}")
        return None, None


def model_exists():
    """Check if a trained model exists in Blob Storage."""
    if not IS_VERCEL or not BLOB_TOKEN:
        return os.path.exists(LOCAL_MODEL_FILE) and os.path.exists(LOCAL_LABELS_FILE)

    try:
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": "model/", "limit": 10},
            timeout=15,
        )
        if resp.status_code == 200:
            blobs = resp.json().get("blobs", [])
            paths = {b.get("pathname") for b in blobs}
            return "model/model.yml" in paths and "model/labels.npy" in paths
        return False
    except Exception as e:
        print(f"Blob model_exists error: {e}")
        return False


def delete_model():
    """Delete model files from Blob Storage."""
    if not IS_VERCEL or not BLOB_TOKEN:
        for f in [LOCAL_MODEL_FILE, LOCAL_LABELS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        return True

    try:
        resp = requests.get(
            BLOB_API_URL,
            headers=_blob_headers(),
            params={"prefix": "model/", "limit": 10},
            timeout=15,
        )
        if resp.status_code != 200:
            return False

        urls = [b.get("url") for b in resp.json().get("blobs", []) if b.get("url")]
        if not urls:
            return True

        del_resp = requests.post(
            f"{BLOB_API_URL}/delete",
            headers={**_blob_headers(), "Content-Type": "application/json"},
            json={"urls": urls},
            timeout=30,
        )
        return del_resp.status_code == 200
    except Exception as e:
        print(f"Blob delete_model error: {e}")
        return False


def get_storage_info():
    """Get debug info about blob storage status."""
    info = {
        "is_vercel": IS_VERCEL,
        "blob_token_set": bool(BLOB_TOKEN),
        "model_exists": model_exists(),
        "people": list_people(),
    }
    total, counts = count_images()
    info["total_images"] = total
    info["image_counts"] = counts
    return info
