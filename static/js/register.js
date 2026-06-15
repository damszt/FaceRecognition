const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const statusDiv = document.getElementById('status');
const progressBar = document.getElementById('progressBar');
const progressContainer = document.querySelector('.progress');

let stream;
const TOTAL_IMAGES = 30;
let capturedCount = 0;
let failedCount = 0;
const MAX_FAILS = 100; // stop if too many consecutive failures

startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        startBtn.disabled = true;
        captureBtn.disabled = false;
        statusDiv.className = 'status-message text-info';
        statusDiv.innerText = "Camera started. Enter name and click Capture Dataset.";
    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusDiv.className = 'status-message text-danger';
        statusDiv.innerText = "Error accessing webcam: " + err.message;
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
    failedCount = 0;
    progressContainer.style.display = 'flex';
    progressBar.style.width = '0%';
    progressBar.innerText = '';
    statusDiv.className = 'status-message text-info';
    statusDiv.innerText = "Capturing images... Position your face clearly in view.";

    captureLoop(name);
});

async function captureLoop(name) {
    // Done!
    if (capturedCount >= TOTAL_IMAGES) {
        statusDiv.className = 'status-message text-success';
        statusDiv.innerText = `✅ Captured ${TOTAL_IMAGES} images for "${name}". Done! You can now Train the model.`;
        captureBtn.disabled = false;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            startBtn.disabled = false;
        }
        return;
    }

    // Too many failures — stop
    if (failedCount >= MAX_FAILS) {
        statusDiv.className = 'status-message text-danger';
        statusDiv.innerText = `❌ Could not detect face after ${MAX_FAILS} attempts. Only captured ${capturedCount}/${TOTAL_IMAGES}. Try better lighting or move closer.`;
        captureBtn.disabled = false;
        return;
    }

    // Capture frame from video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg', 0.9);

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, image: dataURL })
        });

        const result = await response.json();

        if (result.success) {
            capturedCount++;
            failedCount = 0; // reset fail counter on success
            const percent = (capturedCount / TOTAL_IMAGES) * 100;
            progressBar.style.width = percent + '%';
            progressBar.innerText = `${capturedCount}/${TOTAL_IMAGES}`;
            statusDiv.className = 'status-message text-info';
            statusDiv.innerText = `Capturing... ${capturedCount}/${TOTAL_IMAGES} — Move your head slightly for variety.`;
        } else {
            failedCount++;
            // Show reason but don't stop loop
            const reason = result.message || "Face not detected";
            statusDiv.className = 'status-message text-warning';
            statusDiv.innerText = `⚠️ ${reason} (${capturedCount}/${TOTAL_IMAGES} captured, ${failedCount} retries)`;
        }
    } catch (err) {
        failedCount++;
        console.error("Network error:", err);
        statusDiv.className = 'status-message text-danger';
        statusDiv.innerText = `❌ Network error: ${err.message}`;
    }

    // Small delay between captures — 200ms gives server time to respond
    setTimeout(() => captureLoop(name), 200);
}
