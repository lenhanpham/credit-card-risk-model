FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY credit_risk_model/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask gunicorn

# Copy the model code
COPY credit_risk_model /app/credit_risk_model

# Create directories for model artifacts
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "credit_risk_model.src.api.serve_model:app"]