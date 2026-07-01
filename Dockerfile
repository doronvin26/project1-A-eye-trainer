# 1. שימוש בגרסת Python מודרנית (3.11) על בסיס Ubuntu
FROM ubuntu:22.04
# מניעת שאלות אינטראקטיביות בזמן ההתקנה
ENV DEBIAN_FRONTEND=noninteractive

# 2. התקנת Python וספריות מערכת עבור OpenCV
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# יצירת תיקיית מודלים נפרדת (כדי שה-Volume לא ידרוס אותה) והורדת המודל
RUN mkdir -p /models && \
    if [ ! -f /models/pose_landmarker_full.task ]; then \
    wget -q -O /models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task; \
    fi

# הגדרת תיקיית העבודה
WORKDIR /app

# העתקת ה-requirements והתקנה
# (עדיין מעתיקים את זה כי הספריות לא משתנות לעיתים קרובות כמו הקוד)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# הערה: הסרנו את COPY . . כדי להשתמש ב-Mount בזמן ההרצה

# הרצה של Streamlit עם video_labeling_app.py
CMD ["sh", "-c", "echo 'App is starting at: http://localhost:8501' && python3 -m streamlit run knn_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
