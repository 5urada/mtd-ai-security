## VulnBot Setup Guide

### Prerequisites
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### VulnBot Installation

1. **Clone and Setup Environment**
```bash
# Clone VulnBot repository
git clone https://github.com/your-username/VulnBot.git
cd VulnBot

# Create virtual environment with Python 3.11.11
uv venv --python 3.11.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

2. **Database Setup**
```bash
# Install MySQL client and server
sudo apt update
sudo apt install -y mysql-server mysql-client python3-mysqldb

# Start MySQL service
sudo systemctl start mysql
sudo systemctl enable mysql

# Create database and user
sudo mysql -e "CREATE DATABASE vulnbot_db;"
sudo mysql -e "CREATE USER 'vulnbot'@'localhost' IDENTIFIED BY 'vulnbot_pass';"
sudo mysql -e "GRANT ALL PRIVILEGES ON vulnbot_db.* TO 'vulnbot'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"
```

3. **Configuration Files Setup**

Create the following configuration files:

**`basic_config.yaml`**:
```yaml
# Logging settings
log_verbose: true
LOG_PATH: "./logs"

# System mode
mode: "auto"  # Options: auto, manual, semi

# Network settings
http_default_timeout: 30
default_bind_host: "0.0.0.0"

# Kali Linux configuration (attacker node)
hostname: "10.10.1.2"  # Attacker VM IP for CloudLab
port: 22
username: "username"
password: "password"

# Server configuration
api_server:
  host: "0.0.0.0"
  port: 8000
webui_server:
  host: "0.0.0.0" 
  port: 8080
```

**`db_config.yaml`**:
```yaml
mysql:
  host: "localhost"
  port: 3306
  user: "vulnbot"
  password: "vulnbot_pass"
  database: "vulnbot_db"
```

**`model_config.yaml`**:
```yaml
# API settings
api_key: "your_openai_api_key"
base_url: "https://api.openai.com/v1"
llm_model: "openai"
llm_model_name: "gpt-4"

# Embedding settings
embedding_models: "text-embedding-ada-002"
embedding_type: "remote"
context_length: 8192
embedding_url: "https://api.openai.com/v1/embeddings"

# Processing parameters
temperature: 0.1
history_len: 10
timeout: 60
proxies: {}
```

4. **Initialize VulnBot**
```bash
# Create necessary directories
mkdir -p logs data

# Initialize database tables (if script exists)
python cli.py init
```
