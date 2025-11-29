# Face Recognition Web App

This is a Flask-based web application for face recognition attendance.

## Prerequisites

- Python 3.8+
- Webcam

## Getting Started

### 1. Clone the Repository
```bash
git clone <YOUR_REPO_URL>
cd face_auth_app
```

### 2. Set up a Virtual Environment (Recommended)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

1.  Run the application:
    ```bash
    python app.py
    ```
2.  Open your browser and navigate to `http://127.0.0.1:5000`.

## Features

-   **Register**: Capture face images for new users.
-   **Train**: Train the LBPH model with collected images.
-   **Attendance**: Real-time face recognition and attendance logging.
-   **Dashboard**: View daily attendance logs.

## Project Structure

-   `app.py`: Main Flask application.
-   `face_core.py`: Core logic for face recognition.
-   `dataset/`: Stores captured face images.
-   `model.yml`: Trained LBPH model.
-   `labels.npy`: Mapping of label IDs to names.
-   `attendance_YYYY-MM-DD.csv`: Daily attendance logs.
