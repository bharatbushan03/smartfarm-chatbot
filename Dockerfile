# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY main.py .

# Expose the port the app runs on
# Render sets PORT dynamically at runtime.
EXPOSE 10000

# Run the application
# Use PORT if provided (Render), otherwise default to 10000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
