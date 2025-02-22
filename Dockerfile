# Use an official Python runtime as a parent image
FROM python:3.9.7-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD ./data /app/data
ADD ./utils /app/utils
ADD config.json /app
ADD LICENSE /app
ADD main.py /app
ADD README.md /app
ADD requirement.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt
RUN apt-get update && apt-get install -y nano


# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable for Gurobi
ENV GUROBI_HOME=/opt/gurobi
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

# Run app.py when the container launches
# CMD ["python", "main.py"]
CMD [ "/bin/bash" ]
