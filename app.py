from flask import Flask, render_template, request, jsonify
import face_core
import os

app = Flask(__name__)

# Ensure resources are loaded on startup
face_core.load_resources()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

# API Endpoints

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    name = data.get('name')
    image = data.get('image')
    
    if not name or not image:
        return jsonify({"success": False, "message": "Missing name or image data."}), 400
        
    success = face_core.save_face_image(name, image)
    if success:
        return jsonify({"success": True, "message": "Image saved."})
    else:
        return jsonify({"success": False, "message": "Face not detected or save failed."}), 500

@app.route('/api/train', methods=['POST'])
def api_train():
    result = face_core.train_model()
    # Reload resources after training
    face_core.load_resources()
    return jsonify({"message": result})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"error": "No image data"}), 400
        
    name, confidence, details = face_core.recognize_face(image_data)
    
    if name:
        if name != "Unknown":
            # Threshold check (Lower is better for LBPH)
            if confidence < 60:
                logged = face_core.log_attendance(name, confidence, details)
                message = f"Welcome, {name}!"
                if not logged:
                    message = f"Welcome back, {name}! (Already logged)"
                
                return jsonify({
                    "success": True, 
                    "name": name, 
                    "confidence": confidence,
                    "message": message
                })
            else:
                 return jsonify({
                    "success": False, 
                    "name": "Unknown", 
                    "confidence": confidence, 
                    "message": "Face not recognized (Low confidence)."
                })
        else:
            return jsonify({
                "success": False, 
                "name": "Unknown", 
                "confidence": confidence, 
                "message": "Face not recognized."
            })
    else:
        return jsonify({"success": False, "message": "No face detected."})

@app.route('/api/logs', methods=['GET'])
def api_logs():
    date_str = request.args.get('date') # YYYY-MM-DD
    logs = face_core.get_attendance_logs(date_str)
    return jsonify(logs)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    stats = face_core.get_model_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
