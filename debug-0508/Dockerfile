FROM continuumio/miniconda3

# Create a new environment named myenv with Python 3.10
RUN conda create -y -n myenv python=3.9

# Activate it by default when container runs
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install additional packages
# RUN conda install -y numpy scipy

# Set entrypoint
CMD ["python"]
