# Use the official Python image as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirement.txt .

# Install the dependencies
RUN   pip3 install -r requirement.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port number the Flask app runs on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask application
CMD ["flask", "run"]