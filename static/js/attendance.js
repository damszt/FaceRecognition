const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const lastRecDiv = document.getElementById('last-recognition');

let stream;
let intervalId;
let isRecognizing = false;

startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusDiv.innerText = "Camera started. Recognizing...";

        // Start recognition loop
        intervalId = setInterval(recognizeFrame, 2000); // Check every 2 seconds
    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusDiv.innerText = "Error accessing webcam.";
    }
});

stopBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    clearInterval(intervalId);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusDiv.innerText = "Camera stopped.";
});

async function recognizeFrame() {
    if (isRecognizing) return;
    isRecognizing = true;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });

        const result = await response.json();

        if (result.success) {
            lastRecDiv.innerText = result.message;
            lastRecDiv.style.display = 'block';

            // Optional: Speak the name
            if ('speechSynthesis' in window) {
                // Simple debounce for speech could be added here if needed
            }
        } else {
            // statusDiv.innerText = "Scanning...";
        }
    } catch (err) {
        console.error("Error recognizing:", err);
    } finally {
        isRecognizing = false;
    }
}
