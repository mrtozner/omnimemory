#!/usr/bin/env bash
#
# Demo script for OmniMemory Init CLI
#

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║   OmniMemory Init CLI - Demo                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

echo "1. Checking version..."
python3 omni_init.py --version
echo ""

echo "2. Checking tool status..."
python3 omni_init.py status --tool all
echo ""

echo "3. Testing dry-run configuration (Cursor)..."
python3 omni_init.py init --tool cursor --api-key test_key_12345678901234567890 --dry-run
echo ""

echo "4. Showing help..."
python3 omni_init.py --help
echo ""

echo "╔══════════════════════════════════════════════╗"
echo "║   Demo Complete!                             ║"
echo "╚══════════════════════════════════════════════╝"
