FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install system packages first (heavy layer that rarely changes)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    xvfb \
    x11vnc \
    xdotool \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglu1-mesa \
    libopenal1 \
    libvorbis0a \
    libvorbisfile3 \
    libpng16-16 \
    libjpeg8 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    xauth \
    net-tools \
    iputils-ping \
    procps \
    vim \
    fluxbox \
    x11-apps \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Download and install AssaultCube (heavy layer that rarely changes)
RUN wget -O /tmp/assaultcube.tar.bz2 https://github.com/assaultcube/AC/releases/download/v1.3.0.2/AssaultCube_v1.3.0.2_LockdownEdition_RC1.tar.bz2 \
    && cd /tmp \
    && tar -xjf assaultcube.tar.bz2 \
    && mv AssaultCube_v1.3.0.2 /opt/assaultcube \
    && chmod +x /opt/assaultcube/assaultcube.sh \
    && chmod +x /opt/assaultcube/bin_unix/linux_*_client \
    && rm /tmp/assaultcube.tar.bz2

# Install Python packages (medium layer - only changes when requirements change)
COPY requirements.txt /app/requirements.txt
RUN pip3 install gymnasium numpy stable-baselines3 matplotlib opencv-python pyautogui psutil mss torch torchvision Pillow

# Copy application code (light layer - changes frequently)
COPY container_experiments.py /app/
COPY container_entrypoint.py /app/
COPY debug_assaultcube.py /app/

WORKDIR /app

EXPOSE 5900

CMD ["python3", "container_entrypoint.py"]
