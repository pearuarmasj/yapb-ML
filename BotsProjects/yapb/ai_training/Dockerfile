FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV LIBGL_ALWAYS_INDIRECT=1
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

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
    libopenal1 \
    libvorbis0a \
    libvorbisfile3 \
    libpng16-16 \
    libjpeg8 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libopenal-dev \
    libdrm2 \
    libxss1 \
    libgconf-2-4 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0\
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
    && rm -rf /var/lib/apt/lists/*

# Copy user's pre-configured AssaultCube build (light layer - user's local build)
COPY assaultcube-preconfigured/ /opt/assaultcube/
COPY .assaultcube /root/.assaultcube
RUN chmod +x /opt/assaultcube/assaultcube.sh || true \
    && find /opt/assaultcube/bin_unix -name "native_client" -exec chmod +x {} \; 2>/dev/null || true \
    && mkdir -p /opt/assaultcube/logs && chmod 777 /opt/assaultcube/logs

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
