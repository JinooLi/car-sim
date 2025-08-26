# Base image
FROM python:3.13-bookworm

# Install required packages for poetry, sudo and non-root user creation
RUN apt-get update && apt-get install -y \
    curl            \
    sudo            \
    ffmpeg          \   
    imagemagick     \  
    libgl1          \  
    && rm -rf /var/lib/apt/lists/*

# Set the default username
ENV USERNAME="vscode"

# Create a non-root user 'vscode' and add to sudoers group
RUN useradd -m ${USERNAME} && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd && chmod 0440 /etc/sudoers.d/nopasswd

# Set default shell of vscode user to bash
RUN chsh -s /bin/bash ${USERNAME}

# Switch to vscode user
USER ${USERNAME}

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to the PATH (for all shells)
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

# Set the default working directory
WORKDIR /workspace

# Install poetry plugin for using `poetry shell`
RUN poetry self add poetry-plugin-shell

# Set default user to vscode
USER ${USERNAME}