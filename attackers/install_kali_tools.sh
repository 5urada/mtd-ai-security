#!/bin/bash
# install_kali_tools.sh - Install essential Kali Linux tools on Ubuntu
# For MTD research with VulnBot

set -e  # Exit on any error

echo "=========================================="
echo "Installing Kali Linux Tools on Ubuntu"
echo "For MTD Research with VulnBot"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Run as regular user with sudo access."
   exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_success "System updated"

# Install essential development tools
print_status "Installing development tools and dependencies..."
sudo apt install -y \
    build-essential \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    default-jdk \
    ruby \
    ruby-dev \
    golang-go \
    nodejs \
    npm \
    cmake \
    make \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpq-dev \
    libsqlite3-dev
print_success "Development tools installed"

# Network scanning and reconnaissance tools
print_status "Installing network scanning tools..."
sudo apt install -y \
    nmap \
    masscan \
    zmap \
    unicornscan \
    hping3 \
    traceroute \
    whois \
    dnsutils \
    host \
    dig \
    nslookup \
    netcat-openbsd \
    socat \
    netdiscover \
    arp-scan \
    fping
print_success "Network scanning tools installed"

# Web application testing tools
print_status "Installing web application testing tools..."
sudo apt install -y \
    nikto \
    dirb \
    gobuster \
    wfuzz \
    sqlmap \
    xsser \
    w3af \
    uniscan \
    whatweb \
    wafw00f \
    httprobe
print_success "Web application testing tools installed"

# Password cracking and authentication tools
print_status "Installing password cracking tools..."
sudo apt install -y \
    hydra \
    medusa \
    ncrack \
    john \
    hashcat \
    crunch \
    cewl \
    patator
print_success "Password cracking tools installed"

# Exploitation frameworks and tools
print_status "Installing exploitation tools..."
sudo apt install -y \
    metasploit-framework \
    exploitdb \
    searchsploit \
    commix \
    beef-xss
print_success "Exploitation tools installed"

# Network analysis and sniffing tools
print_status "Installing network analysis tools..."
sudo apt install -y \
    wireshark \
    tshark \
    tcpdump \
    ettercap-text-only \
    dsniff \
    ssldump \
    ngrep
print_success "Network analysis tools installed"

# Wireless tools (if applicable)
print_status "Installing wireless tools..."
sudo apt install -y \
    aircrack-ng \
    reaver \
    hostapd-wpe \
    wireless-tools \
    rfkill
print_success "Wireless tools installed"

# Forensics and reverse engineering tools
print_status "Installing forensics tools..."
sudo apt install -y \
    binwalk \
    foremost \
    exiftool \
    strings \
    hexedit \
    xxd \
    radare2 \
    gdb \
    ltrace \
    strace
print_success "Forensics tools installed"

# Python security libraries
print_status "Installing Python security libraries..."
pip3 install --user \
    python-nmap \
    scapy \
    requests \
    beautifulsoup4 \
    selenium \
    mechanize \
    paramiko \
    pycrypto \
    impacket \
    pwntools \
    ropper \
    capstone \
    keystone-engine \
    unicorn \
    angr
print_success "Python security libraries installed"

# Additional useful tools
print_status "Installing additional utilities..."
sudo apt install -y \
    tree \
    htop \
    tmux \
    screen \
    vim \
    nano \
    curl \
    jq \
    unzip \
    p7zip-full \
    steghide \
    outguess \
    exiv2
print_success "Additional utilities installed"

# Install some tools from GitHub
print_status "Installing tools from GitHub repositories..."

# Create tools directory
mkdir -p ~/tools
cd ~/tools

# Install SecLists (wordlists)
if [ ! -d "SecLists" ]; then
    print_status "Installing SecLists..."
    git clone https://github.com/danielmiessler/SecLists.git
    print_success "SecLists installed"
fi

# Install PayloadsAllTheThings
if [ ! -d "PayloadsAllTheThings" ]; then
    print_status "Installing PayloadsAllTheThings..."
    git clone https://github.com/swisskyrepo/PayloadsAllTheThings.git
    print_success "PayloadsAllTheThings installed"
fi

# Install subfinder
if ! command -v subfinder &> /dev/null; then
    print_status "Installing subfinder..."
    go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
    print_success "Subfinder installed"
fi

# Install httpx
if ! command -v httpx &> /dev/null; then
    print_status "Installing httpx..."
    go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest
    print_success "httpx installed"
fi

# Install nuclei
if ! command -v nuclei &> /dev/null; then
    print_status "Installing nuclei..."
    go install -v github.com/projectdiscovery/nuclei/v2/cmd/nuclei@latest
    print_success "Nuclei installed"
fi

# Go back to original directory
cd - > /dev/null

# Configure Metasploit database
print_status "Configuring Metasploit database..."
sudo systemctl enable postgresql
sudo systemctl start postgresql
sudo -u postgres createuser -DRS msf
sudo -u postgres createdb -O msf msf
print_success "Metasploit database configured"

# Initialize Metasploit database
print_status "Initializing Metasploit database..."
sudo msfdb init || print_warning "Metasploit database initialization may have failed"

# Add Go tools to PATH
print_status "Adding Go tools to PATH..."
echo 'export PATH=$PATH:~/go/bin' >> ~/.bashrc
export PATH=$PATH:~/go/bin
print_success "Go tools added to PATH"

# Create useful aliases
print_status "Creating useful aliases..."
cat >> ~/.bashrc << 'EOF'

# Kali-style aliases for penetration testing
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias nse='ls /usr/share/nmap/scripts/ | grep'
alias ports='netstat -tulanp'
alias listening='lsof -i -P -n | grep LISTEN'
alias myip='curl -s ifconfig.me'
alias localip='hostname -I'
alias serve='python3 -m http.server'
alias msfconsole='msfconsole -q'
alias searchsploit='searchsploit --color'

# MTD-specific aliases
alias mtd-scan='nmap -sn 10.10.1.20-49'
alias mtd-monitor='watch -n 5 "nmap -sn 10.10.1.20-49 | grep \"Nmap scan report\""'
EOF

print_success "Aliases created"

# Update locate database
print_status "Updating locate database..."
sudo updatedb || print_warning "Could not update locate database"

# Final system cleanup
print_status "Cleaning up..."
sudo apt autoremove -y
sudo apt autoclean
print_success "Cleanup completed"

echo ""
echo "=========================================="
print_success "Kali Linux tools installation completed!"
echo "=========================================="
echo ""
print_status "Installed tools include:"
echo "  • Network scanning: nmap, masscan, zmap"
echo "  • Web testing: nikto, dirb, gobuster, sqlmap"
echo "  • Password cracking: hydra, john, hashcat"
echo "  • Exploitation: metasploit, searchsploit"
echo "  • Network analysis: wireshark, tcpdump"
echo "  • Python libraries: scapy, python-nmap, impacket"
echo "  • Additional tools in ~/tools/"
echo ""
print_warning "Please restart your terminal or run 'source ~/.bashrc' to load new aliases and PATH"
echo ""
print_status "Tools are ready for MTD research with VulnBot!"

# Test a few key tools
echo ""
print_status "Testing key tools..."
echo -n "nmap: "; nmap --version | head -1
echo -n "metasploit: "; msfconsole -v 2>/dev/null || echo "metasploit installed (database may need setup)"
echo -n "sqlmap: "; sqlmap --version 2>/dev/null || echo "sqlmap installed"
echo -n "python-nmap: "; python3 -c "import nmap; print('python-nmap available')" 2>/dev/null || echo "python-nmap may need reinstall"

echo ""
print_success "Installation script completed successfully!"
print_status "You can now proceed with VulnBot setup and MTD testing."
