# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code and install dependencies
COPY ../requirements.txt .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
