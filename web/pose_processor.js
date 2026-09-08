// web/pose_processor.js

let poseLandmarker = undefined;
const video = document.getElementById("input-video");
const canvas = document.getElementById("output-canvas");
const ctx = canvas.getContext("2d");

// נחכה שהספרייה תטען מה-window (כי היא גלובלית ב-CDN)
async function initMediaPipe() {
    // MediaPipe חושף את האובייקט vision דרך ה-bundle שטענו ב-HTML
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );
    
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO"
    });
    
    startCamera();
}

// וודא ש-FilesetResolver קיים ב-window (הוא מגיע מה-bundle)
const { PoseLandmarker, FilesetResolver } = window.vision; 

initMediaPipe();