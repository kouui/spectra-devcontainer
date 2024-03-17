ARG PYVER=3.11
FROM python:${PYVER}-slim

# Install necessary tools (need root)
#RUN apt-get update && apt-get install -y gcc git curl python3.10 python3.10-distutils python3.10-venv python3-pip fish
RUN apt-get update && apt install -y git curl fish sudo

# Optionally, set fish as the default shell (need root)
#RUN chsh -s /usr/bin/fish

# Clean up (need root)
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a user (in docker group and sodoers)
ARG USER=dev
ARG GROUP=docker
ARG PASSWORD=docker
ARG GID=1000
ARG UID=1000
RUN groupadd --gid ${GID} ${GROUP} && \
    useradd --uid ${UID} --gid ${GID} --create-home ${USER} && \
    adduser ${USER} sudo && \
    echo "${USER}:${PASSWORD}" | chpasswd
USER ${USER}

# # Create a group with GID 1000
# RUN groupadd -g 1000 devgroup
# # add user
# RUN useradd -m -d /home/dev -s /bin/bash -u 1000 -g 1000 dev


# Create a virtual environment
RUN python${PYVER} -m venv /home/${USER}/venv

# Activate the virtual environment and install the packages
# RUN /bin/bash -c "source /home/${USER}}/venv/bin/activate && \
#     pip install numpy==1.25.2 scikit-learn==1.3.0 && \
#     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

# create working directory to mount the current directory in WSL
#RUN mkdir /workspace (since we nolonger as root user, not able to create /workspace)

# Install Poetry
#RUN curl -sSL https://install.python-poetry.org | python3 -

# Set environment variable for poetry
#ENV POETRY_VIRTUALENVS_CREATE=false

# Install Python packages
# NOTE: You might need additional steps to properly install pytorch with CUDA. 
# This is a placeholder, and you should consult the official PyTorch installation guide.
#RUN poetry install numpy scikit-learn
#RUN peotry install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# setup git
#RUN git config --global user.email "huang.yuwei@plus-zero.co.jp"
#RUN git config --global user.name "huang.yuwei"

#SHELL ["/usr/bin/fish", "-c"]

# setup command for activating venv
# RUN mkdir -p /home/${USER}/.config/fish && echo "alias activate='source /home/${USER}/venv/bin/activate.fish'" >> /home/${USER}/.config/fish/config.fish
# RUN echo "activate" >> /home/${USER}/.config/fish/config.fish
RUN echo "alias activate='source /home/${USER}}/venv/bin/activate'" >> /home/${USER}/.bashrc
RUN echo "activate" >> /home/${USER}/.bashrc
#CMD ["echo", "alias activate='source /home/dev/venv/bin/activate.fish'",  ">>",  "~/.bashrc"]

# keep container alive
#CMD ["tail", "-f", "/dev/null"]

ENTRYPOINT ["/bin/sh", "-c", "sleep infinity"]
