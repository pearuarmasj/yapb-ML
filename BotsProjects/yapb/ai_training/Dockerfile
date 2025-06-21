FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV LIBGL_ALWAYS_INDIRECT=0
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=2.1
ENV MESA_GLSL_VERSION_OVERRIDE=120
ENV GALLIUM_DRIVER=llvmpipe
ENV LP_NUM_THREADS=1

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
    libogg-dev \
    libvorbis-dev \
    libdrm2 \
    libxss1 \
    libgconf-2-4 \
    libxrandr2 \
    libasound2 \
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
    && rm -rf /var/lib/apt/lists/*

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

WORKDIR /app

EXPOSE 5900

CMD ["python3", "container_entrypoint.py"]
