FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libyaml-dev \
    ffmpeg \
    swig \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash user

WORKDIR /home/user

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir jax[cpu]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || true && \
    pip3 install jax==0.4.13 || true && \
    pip3 install -r requirements.txt

COPY . .

RUN mkdir -p ~/ogbench_data

RUN chmod +x /home/user/hilp_gcrl/*.sh 2>/dev/null || true

ENV MUJOCO_KEY_PATH=/home/user/.mujoco

USER user

CMD ["/bin/bash"]