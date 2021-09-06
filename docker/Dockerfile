FROM nvidia/cudagl:11.1-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        vim git curl ca-certificates zip unzip build-essential cmake pkg-config \
        python3-dev python3-pip python3-setuptools python3-wheel python-is-python3 \
        libjpeg-dev libglm-dev libgl1-mesa-glx libgl1-mesa-dev mesa-utils \
        xorg-dev freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy~=1.19.2 numpy-quaternion==2020.11.2.17.0.49 tensorflow && \
    pip3 install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html

ARG HABITAT_SIM_VERSION=v0.1.7
RUN git clone https://github.com/facebookresearch/habitat-sim --branch ${HABITAT_SIM_VERSION} && \
    cd habitat-sim && \
    pip3 install --no-cache-dir -r requirements.txt && \
    python3 setup.py build_ext --parallel 6 install --headless --with-cuda

ADD . /habitat-lab
RUN cd habitat-lab && \
    pip3 install --no-cache-dir -r requirements.txt && \
    python3 setup.py develop --all

ENV GLOG_minloglevel=2 MAGNUM_LOG="quiet"

ENTRYPOINT ["python3", "/habitat-lab/scripts/seq_objnav_rpc_eval/run_benchmark.py"]
CMD []
