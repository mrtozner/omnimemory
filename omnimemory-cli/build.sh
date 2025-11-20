#!/bin/bash
# Build script for OmniMemory CLI

echo "Building OmniMemory CLI..."

# Try to install Rust if not available
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Check the project structure
echo "Checking project structure..."
find src -name "*.rs" | head -10

echo "Running cargo check..."
cargo check --message-format=short

echo "If check passes, run 'cargo build --release' to build the binary"