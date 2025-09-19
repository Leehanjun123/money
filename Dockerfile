FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set environment variables
ENV DJANGO_SETTINGS_MODULE=stylemate.settings
ENV PYTHONPATH=/app

# Make start script executable
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Run the application
CMD ["./start.sh"]