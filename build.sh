#!/bin/bash

# Configuration
PYTHON_VERSION="3.10.12"  # Or a widely supported version
VENV_NAME="dp_finetune_env"
REPO_URL="https://github.com/saranggalada/Differentially-private-LLM-finetuning.git" # Replace with your repo URL
SCRIPT_NAME="dp-llm-finetune.py"
CONFIG_FILE="dp-finetune-config.json" # Name of your config file

# Check Python version
MIN_PYTHON_VERSION="3.8"
CURRENT_PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1-2)
# Function to compare version numbers (major.minor)
compare_versions() {
    local v1="$1"
    local v2="$2"
    local v1_major=$(echo "$v1" | cut -d'.' -f1)
    local v1_minor=$(echo "$v1" | cut -d'.' -f2)
    local v2_major=$(echo "$v2" | cut -d'.' -f1)
    local v2_minor=$(echo "$v2" | cut -d'.' -f2)

    if [[ "$v1_major" -gt "$v2_major" ]]; then
        echo 1 # v1 > v2
        return
    elif [[ "$v1_major" -lt "$v2_major" ]]; then
        echo -1 # v1 < v2
        return
    else
        if [[ "$v1_minor" -gt "$v2_minor" ]]; then
            echo 1 # v1 > v2
            return
        elif [[ "$v1_minor" -lt "$v2_minor" ]]; then
            echo -1 # v1 < v2
            return
        else
            echo 0 # v1 == v2
            return
        fi
    fi
}

version_comparison=$(compare_versions "$CURRENT_PYTHON_VERSION" "$MIN_PYTHON_VERSION")

if [[ "$version_comparison" -ge 0 ]]; then # Current version >= minimum required
    echo "Python version $CURRENT_PYTHON_VERSION is sufficient (>= $MIN_PYTHON_VERSION). Skipping installation."
    PYTHON_EXECUTABLE="python3" # Use system python3
else
    echo "Python version $CURRENT_PYTHON_VERSION is older than $MIN_PYTHON_VERSION. Installing Python $PYTHON_VERSION..."
    # Install Python (same installation code as before)
    sudo apt-get update
    sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
    tar -xf "Python-${PYTHON_VERSION}.tgz"
    cd "Python-${PYTHON_VERSION}"
    ./configure --enable-optimizations
    make -j$(nproc)
    sudo make altinstall
    cd ..
    rm -rf "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz"
    echo "Python $PYTHON_VERSION installed successfully."
    PYTHON_EXECUTABLE="python${PYTHON_VERSION}" # Use installed python
fi

# Clone/Update repository
if [ ! -d "dp-llm-finetune-project" ]; then
    git clone "$REPO_URL" dp-llm-finetune-project
else
    cd dp-llm-finetune-project
    git pull origin main # Or your branch
    cd ..
fi

cd dp-llm-finetune-project

# Create/Activate virtual environment (using the correct python executable)
if [ ! -d "$VENV_NAME" ]; then
    "$PYTHON_EXECUTABLE" -m venv "$VENV_NAME"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

source "$VENV_NAME/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Run the Python script with config file as argument
python "$SCRIPT_NAME" --config "$CONFIG_FILE"

# Deactivate
deactivate

cd ..

echo "Finetuning process complete."