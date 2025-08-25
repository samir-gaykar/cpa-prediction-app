FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy model & pipeline files
COPY model.joblib \
     data_loader.pkl \
     temporal_feature_engineer_transformer.pkl \
     drop_columns_transformer.pkl \
     log_transformer.pkl \
     boolean_to_string_transformer.pkl \
     target_encoder_transformer.pkl \
     standard_scaler_transformer.pkl \
     interaction_transformer.pkl \
     column_selector.pkl \
     inference_pipeline.pkl \
     ./

# Copy all custom transformers
COPY data_loader.py \
     temporal_feature_engineer_transformer.py \
     drop_columns_transformer.py \
     log_transformer.py \
     boolean_to_string_transformer.py \
     target_encoder_transformer.py \
     standard_scaler_transformer.py \
     interaction_transformer.py \
     column_selector.py \
     ./

# Copy application code
COPY server.py \
     app.py \
     requirements.txt \
     ./
     
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install system dependencies (supervisor)
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Supervisor logs directory
RUN mkdir -p /var/log/supervisor

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Start both FastAPI and Streamlit via Supervisor
CMD ["/usr/bin/supervisord", "-n"]