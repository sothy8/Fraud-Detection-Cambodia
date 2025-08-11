#!/bin/bash

# Fraud Detection Cambodia - Cleanup Script
# Removes unnecessary files and cache

echo "🧹 Cleaning up Fraud Detection Cambodia project..."

# Remove system files
echo "🗑️  Removing system files..."
find . -name ".DS_Store" -type f -delete 2>/dev/null || true
find . -name "Thumbs.db" -type f -delete 2>/dev/null || true
find . -name "*.tmp" -type f -delete 2>/dev/null || true

# Remove Python cache
echo "🐍 Removing Python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove IDE files
echo "💻 Removing IDE files..."
find . -name ".vscode" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".idea" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.swp" -delete 2>/dev/null || true
find . -name "*.swo" -delete 2>/dev/null || true

# Remove temporary data files
echo "📊 Cleaning temporary data files..."
find data/datasets -name "*.csv" -mtime +30 -delete 2>/dev/null || true  # Remove datasets older than 30 days
find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true  # Remove logs older than 7 days

# Remove empty directories
echo "📁 Removing empty directories..."
find . -type d -empty -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null || true

echo "✅ Cleanup completed!"

# Show current project size
echo ""
echo "📏 Current project size:"
du -sh . 2>/dev/null || echo "Size calculation not available"

echo ""
echo "📁 Remaining structure:"
find . -maxdepth 2 -type d -not -path "./venv*" -not -path "./.git*" | sort
