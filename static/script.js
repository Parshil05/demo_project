const video = document.getElementById('video');
const resultDiv = document.getElementById('result');

// ✅ Start the camera (works on mobile & desktop)
navigator.mediaDevices.getUserMedia({ video: { facingMode: ["user", "environment"] } })
    .then(stream => { 
        video.srcObject = stream; 
        console.log("Camera access granted.");
    })
    .catch(err => {
        console.error("Camera access denied:", err);
        alert("Please allow camera access to use this feature.");
    });

// ✅ Function to capture and send frame every 2 seconds
function captureAndProcess() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg', 0.7); // Adjust quality if needed

    fetch('/process_frame', {  
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.innerHTML = "<h2>Processing Result:</h2>";

        if (!data.faces || data.faces.length === 0) {  // ✅ Check if faces exist
            resultDiv.innerHTML = "<p>No face detected. Please try again.</p>";
            return;
        }

        data.faces.forEach(face => {
            resultDiv.innerHTML += `<p>Detected: ${face.label} (Similarity: ${face.similarity.toFixed(2)})</p>`;
        });
    })
    .catch(err => console.error("Processing error:", err));
}

// ✅ Auto-capture every 2 seconds
setInterval(captureAndProcess, 2000);
