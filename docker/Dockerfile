FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36

USER root
RUN apt-get update && apt-get -y upgrade
RUN apt-get install git
RUN apt-get install -y \
    curl \
    gdb \
    htop \
    && rm -rf /var/lib/apt/lists/*
USER user

RUN pip install --upgrade pip
COPY requirements.txt /home/jovyan/
RUN pip install --no-cache-dir -r /home/jovyan/requirements.txt

# Install AIJ Multiagent AI
RUN pip install git+https://github.com/AIRI-Institute/aij_multiagent_rl.git

ENV PYTHONWARNINGS ignore::DeprecationWarning
