FROM python:3.10.17-alpine3.20
# FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# RUN apk add --no-cache --virtual .build-deps gcc musl-dev \
#     && pip install --no-cache-dir --upgrade pip \
#     && apk del .build-deps

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8001

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]