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

# Run the application using gunicorn for production
# Use 2 workers for free tier compatibility (512MB RAM)
CMD ["sh", "-c", "gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT:-10000}"]
