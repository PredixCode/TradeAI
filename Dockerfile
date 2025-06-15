# Use the official TensorFlow image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy your project's requirements file
COPY requirements.txt .

# Install your Python dependencies
# (You would create a requirements.txt file with pandas, yfinance, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
COPY . .

# Command to run when the container starts
CMD ["python", "main.py"]