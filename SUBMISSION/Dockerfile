# SIG challenge agent docker image for submission

FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        python3 python3-pip python3-setuptools python3-wheel && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir numpy gym && \
    python3 -m pip install --no-cache-dir torch==1.9.0+cu111 \
                                          torchvision==0.10.0+cu111 \
                           -f https://download.pytorch.org/whl/torch_stable.html

ADD ./SUBMISSION /SUBMISSION/
ADD ./habitat/utils/rpc_evaluation_server.py /habitat/utils/
ADD ./scripts/seq_objnav_rpc_eval/run_agent.py /
ADD ./scripts/seq_objnav_rpc_eval/entry_point_agent.sh /

ENTRYPOINT ["/bin/bash", "/entry_point_agent.sh"]
CMD []
