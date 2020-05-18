FROM rapidsai/rapidsai:0.10-cuda10.1-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    xvfb \
    libcairo2-dev

# Create working directory to add repo.
WORKDIR /workshop

# Load contents into working directory.
ADD . .

# Do not chain these commands
RUN conda run -n rapids pip install --upgrade pip
RUN conda run -n rapids pip install pyvirtualdisplay
RUN conda run -n rapids pip install ray==0.8.0
RUN conda run -n rapids pip install ray[rllib]
RUN conda run -n rapids pip install tensorflow==2.1.0
RUN conda run -n rapids pip install flatland-rl==2.1.10
RUN conda run -n rapids pip install jupyterlab
RUN conda run -n rapids jupyter lab build

# Create working directory.
WORKDIR /workshop/nvidia-gtc-instadeep

# Jupyter listens on 8888.
EXPOSE 8888

# Please see `entrypoint.sh` for details on how this content
# is launched.
ADD entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
