FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV SDL_AUDIODRIVER=dummy
ENV ALSA_DISABLE=1
ENV PULSE_DISABLE=1
ENV OPENAL_DISABLE=1
ENV PULSE_RUNTIME_PATH=/dev/null
ENV PULSE_CONFIG_PATH=/dev/null
ENV ALSA_CONFIG_PATH=/dev/null
ENV ALSA_MIXER_SIMPLE=1
ENV SDL_AUDIO_DRIVER=dummy
ENV AUDIODEV=/dev/null

# Install system packages first (heavy layer that rarely changes)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    xvfb \
    x11vnc \
    xdotool \
    wget \
    zip \
    libgl1-mesa-glx \
    libglu1-mesa \
    mesa-utils \
    mesa-utils-extra \
    libpng16-16 \
    libjpeg8 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-dev \
    libsdl2-image-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libogg-dev \
    libvorbis-dev \
    libopenal-dev \
    libdrm2 \
    libxss1 \
    libgconf-2-4 \
    libxrandr2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    xauth \
    net-tools \
    iputils-ping \
    procps \
    vim \
    fluxbox \
    x11-apps \
    imagemagick \
    x11-xserver-utils \
    x11-utils \
    xinit \
    libxkbcommon-x11-0 \
    libx11-dev \
    libxext-dev \
    libxtst6 \
    binutils \
    gdb \
    rsync \
    socat \
    cmake \
    git \
    g++ \
    gcc \
    build-essential \
    clang \
    make \
    ninja-build \
    autotools-dev \
    autoconf \
    libtool \
    file \
    && rm -rf /var/lib/apt/lists/*

# Create dummy audio configuration to prevent ALSA/OpenAL initialization errors
RUN mkdir -p /root/.config/pulse /root/.asoundrc.d /etc/openal && \
    echo "autospawn = no" > /root/.config/pulse/client.conf && \
    echo "daemon-binary = /bin/true" >> /root/.config/pulse/client.conf && \
    echo "pcm.!default { type plug; slave { pcm null } }" > /root/.asoundrc && \
    echo "ctl.!default { type hw; card 0 }" >> /root/.asoundrc && \
    echo "pcm.null { type null }" >> /root/.asoundrc && \
    echo "ctl.null { type null }" >> /root/.asoundrc && \
    echo "[general]" > /etc/openal/alsoft.conf && \
    echo "drivers = null" >> /etc/openal/alsoft.conf && \
    echo "frequency = 44100" >> /etc/openal/alsoft.conf

# Build AssaultCube from source
COPY .assaultcube /root/.assaultcube
COPY build_assaultcube.sh /tmp/build_assaultcube.sh
RUN chmod +x /tmp/build_assaultcube.sh && /tmp/build_assaultcube.sh && rm /tmp/build_assaultcube.sh

# Install Python packages (medium layer - only changes when requirements change)
COPY requirements.txt /app/requirements.txt
RUN pip3 install gymnasium numpy stable-baselines3 matplotlib opencv-python pyautogui psutil mss torch torchvision Pillow

# Copy application code (light layer - changes frequently)
COPY container_experiments.py /app/
COPY container_entrypoint.py /app/
COPY debug_assaultcube.py /app/

# Create a wrapper script to launch AssaultCube with complete audio disabling
RUN echo '#!/bin/bash' > /app/launch_assaultcube.sh && \
    echo 'export SDL_AUDIODRIVER=dummy' >> /app/launch_assaultcube.sh && \
    echo 'export ALSA_DISABLE=1' >> /app/launch_assaultcube.sh && \
    echo 'export PULSE_DISABLE=1' >> /app/launch_assaultcube.sh && \
    echo 'export OPENAL_DISABLE=1' >> /app/launch_assaultcube.sh && \
    echo 'export PULSE_RUNTIME_PATH=/dev/null' >> /app/launch_assaultcube.sh && \
    echo 'export ALSA_CONFIG_PATH=/dev/null' >> /app/launch_assaultcube.sh && \
    echo 'export AUDIODEV=/dev/null' >> /app/launch_assaultcube.sh && \
    echo 'unset PULSE_RUNTIME_PATH' >> /app/launch_assaultcube.sh && \
    echo 'exec "$@"' >> /app/launch_assaultcube.sh && \
    chmod +x /app/launch_assaultcube.sh

WORKDIR /app

EXPOSE 5900

CMD ["python3", "container_entrypoint.py"]
