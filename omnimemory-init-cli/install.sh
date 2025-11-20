#!/usr/bin/env bash
#
# OmniMemory Init CLI - Installation Script
#
# Usage: ./install.sh
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════╗"
    echo "║   OmniMemory Init CLI Installer           ║"
    echo "║   Configure AI Tools Automatically        ║"
    echo "╚════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        print_info "Install Python 3.8+ from https://python.org"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python ${PYTHON_VERSION} found"
}

# Check pip
check_pip() {
    if ! python3 -m pip --version &> /dev/null; then
        print_error "pip not found"
        print_info "Install pip: curl https://bootstrap.pypa.io/get-pip.py | python3"
        exit 1
    fi
    print_success "pip found"
}

# Install package
install_package() {
    print_info "Installing OmniMemory Init CLI..."

    if python3 -m pip install -e . > /dev/null 2>&1; then
        print_success "Installation complete"
    else
        print_error "Installation failed"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    if python3 omni_init.py --version > /dev/null 2>&1; then
        print_success "Installation verified"
    else
        print_warning "Installation succeeded but CLI not in PATH"
        print_info "You can run the CLI with: python3 omni_init.py"
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}Installation Complete!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Set your API key:"
    echo "   export OMNIMEMORY_API_KEY=\"your_key_here\""
    echo ""
    echo "2. Check tool status:"
    echo "   python3 omni_init.py status --tool all"
    echo ""
    echo "3. Configure a tool:"
    echo "   python3 omni_init.py init --tool claude"
    echo ""
    echo "For more examples, see: USAGE_EXAMPLES.md"
    echo ""
}

# Main
main() {
    print_header

    check_python
    check_pip
    install_package
    verify_installation
    print_next_steps
}

main
