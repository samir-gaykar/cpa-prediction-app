FROM python:3.10-slim

# Install system dependencies (supervisor)
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy everything except what's in .dockerignore
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Supervisor logs directory
RUN mkdir -p /var/log/supervisor

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Start both FastAPI and Streamlit via Supervisor
CMD ["/usr/bin/supervisord", "-n"]