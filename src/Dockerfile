FROM python:3.10-slim

# Install system dependencies (supervisor) first
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Set the working directory for the app
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code except what's in .dockerignore
COPY . .

# Create Supervisor logs directory
RUN mkdir -p /var/log/supervisor

# Copy Supervisor config to default path
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose application ports
EXPOSE 8000 8501

# Start FastAPI and Streamlit via Supervisor
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]