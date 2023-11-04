FROM ubuntu:22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt update && apt upgrade -y &&\
    apt install -y \
    git \
    wget \
    unzip \
    python3-pip \ 
    cmake \
    libeigen3-dev \
    x11-xserver-utils \
    gnuplot \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/alandefreitas/matplotplusplus \
    && cd matplotplusplus \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2" -DMATPLOTPP_BUILD_EXAMPLES=OFF -DMATPLOTPP_BUILD_TESTS=OFF \
    && cmake --build . --parallel 2 --config Release \
    && cmake --install .
