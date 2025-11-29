const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const statusDiv = document.getElementById('status');
const progressBar = document.getElementById('progressBar');
const progressContainer = document.querySelector('.progress');

let stream;
const TOTAL_IMAGES = 50;
let capturedCount = 0;

startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        startBtn.disabled = true;
        captureBtn.disabled = false;
        statusDiv.innerText = "Camera started. Enter name and click Capture.";
    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusDiv.innerText = "Error accessing webcam. Please allow permissions.";
    }
});

captureBtn.addEventListener('click', async () => {
    const name = document.getElementById('personName').value.trim();
    if (!name) {
        alert("Please enter a name.");
        return;
    }
    
    captureBtn.disabled = true;
    capturedCount = 0;
    progressContainer.style.display = 'flex';
    statusDiv.innerText = "Capturing images...";
    
    captureLoop(name);
});

async function captureLoop(name) {
    if (capturedCount >= TOTAL_IMAGES) {
        statusDiv.innerText = `Captured ${TOTAL_IMAGES} images for ${name}. Done!`;
        statusDiv.classList.remove('text-info');
        statusDiv.classList.add('text-success');
        captureBtn.disabled = false;
        
        // Stop camera
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            startBtn.disabled = false;
        }
        return;
    }
    
    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');
    
    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, image: dataURL })
        });
        
        const result = await response.json();
        if (result.success) {
            capturedCount++;
            const percent = (capturedCount / TOTAL_IMAGES) * 100;
            progressBar.style.width = percent + '%';
            progressBar.innerText = `${capturedCount}/${TOTAL_IMAGES}`;
        } else {
            console.warn("Face not detected or save failed.");
        }
    } catch (err) {
        console.error("Error sending frame:", err);
    }
    
    // Delay slightly to avoid overwhelming server and allow movement
    setTimeout(() => captureLoop(name), 100);
}
