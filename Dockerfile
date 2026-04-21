# Use official Python image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 7860

# Run Flask app
CMD ["python", "medichat.py"]