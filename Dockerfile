# Use the official TensorFlow image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy your project's requirements file
COPY requirements.txt .

# Install your Python dependencies
# (You would create a requirements.txt file with pandas, yfinance, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# --- FIX for pandas-ta and numpy>=2.0 ---
# The pandas-ta library uses 'np.NaN' which was removed in numpy 2.0.
# This command finds the problematic file and replaces 'NaN' with 'nan'.
RUN sed -i 's/from numpy import NaN as npNaN/from numpy import nan as npNaN/g' /usr/local/lib/python3.11/dist-packages/pandas_ta/momentum/squeeze_pro.py

# Copy the rest of your project code into the container
COPY . .

# Command to run when the container starts
ENTRYPOINT ["python", "main.py"]