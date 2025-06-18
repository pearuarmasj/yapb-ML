FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

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
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/assaultcube.tar.bz2 https://github.com/assaultcube/AC/releases/download/v1.3.0.2/AssaultCube_v1.3.0.2_LockdownEdition_RC1.tar.bz2 \
    && mkdir -p /opt/assaultcube \
    && cd /opt/assaultcube \
    && tar -xjf /tmp/assaultcube.tar.bz2 \
    && rm /tmp/assaultcube.tar.bz2

COPY requirements.txt /app/requirements.txt
RUN pip3 install gymnasium numpy stable-baselines3 matplotlib opencv-python pyautogui psutil mss torch torchvision

COPY container_experiments.py /app/
COPY container_entrypoint.py /app/

WORKDIR /app

EXPOSE 5900

CMD ["python3", "container_entrypoint.py"]
