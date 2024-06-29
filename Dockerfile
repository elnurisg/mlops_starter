# In the root run the following commands: build the image and run the container
# docker build -t mlops_starter_api .
# docker run -p 8000:8000 mlops_starter_api

# port 8000 is mapped on host machine to port 8000 on the container. 
# Access the FastAPI application by navigating to http://localhost:8000 in web browser.



# this base image should be enough because it includes a minimal Debian-based Linux distribution tailored for running Python
FROM python:3.11-slim

# We set the working directory in the container
WORKDIR /app

# Copy the package requirements and setup files
COPY src/ml_package/requirements.txt src/ml_package/setup.py /app/src/ml_package/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r /app/src/ml_package/requirements.txt

# Copy the package source code
COPY src/ml_package /app/src/ml_package

# Install the package
RUN pip install /app/src/ml_package

# Copy the rest of the working directory contents into the container at /app
COPY . /app/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "src.ml_package.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
