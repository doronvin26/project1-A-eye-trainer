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
    && rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית העבודה
WORKDIR /app

# העתקת ה-requirements והתקנה
# (עדיין מעתיקים את זה כי הספריות לא משתנות לעיתים קרובות כמו הקוד)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# הערה: הסרנו את COPY . . כדי להשתמש ב-Mount בזמן ההרצה

#EXPOSE 8501

# הרצה של Streamlit (שים לב לשימוש ב-python3)
CMD ["sh", "-c", "echo 'App is starting at: http://localhost:8501' && python3 -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --logger.level=error > /dev/null 2>&1"]