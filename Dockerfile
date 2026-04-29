# Use a lightweight Python image
FROM python:3.9-slim

# Install system tools needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Open the port for your web interface
EXPOSE 8501

# Run the application
CMD ["sh", "-c", "echo 'App is starting at: http://localhost:8501' && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --logger.level=error > /dev/null 2>&1"]