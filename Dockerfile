FROM python:3.12.3-slim

# Install Streamlit
RUN pip install streamlit

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app