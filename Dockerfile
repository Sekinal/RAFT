FROM ghcr.io/prefix-dev/pixi:0.40.0

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspaceraft

# Copy pixi.toml and pixi.lock first to leverage Docker cache for dependencies
COPY pixi.toml pixi.lock ./

# Install Python dependencies via pixi, cached unless pixi.toml or pixi.lock change
RUN pixi install --locked

# Copy the source code after dependencies for better caching
COPY . .

RUN chmod +x download_models.sh && \
    pixi run bash download_models.sh || echo "Models will be downloaded at runtime"

# Create necessary directories
RUN mkdir -p datasets outputs runs

# Set permissions for all files inside the workspace
RUN chmod -R 777 /workspaceraft

# Default command to start the pixi shell for interaction
CMD ["pixi", "shell"]
