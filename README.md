# project1-A-eye-trainer
# Overview:
The A-EYE TRAINER is an innovative AI-powered personal trainer application designed to improve the physical training experience. By utilizing computer vision and real-time motion analysis, the system identifies body structure during exercise to provide precise corrective feedback and accurate repetition counting.  The project addresses the common issues of incorrect exercise form, which can lead to injuries and frustration, by making professional-grade coaching accessible to anyone with a smartphone and a computer.  
## 🛠 System Architecture

The application operates using a **distributed computing approach**:

---

### 📱 Frontend
A mobile application that captures the user's movement via the phone's camera.

### 💻 Backend
A computer that receives data from the phone to perform complex algorithmic analysis and pose estimation.

### 🧠 Logic
A state machine-based system manages the exercise logic to:
*   **Track progress** across different sets.
*   **Count repetitions** accurately (e.g., push-ups, planks, and squats).
*   **Provide feedback** based on movement quality.

--- 
