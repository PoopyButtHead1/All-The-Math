#!/bin/bash

# Financial Analysis Suite Setup Script

echo "🚀 Financial Analysis Suite - Setup"
echo "====================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "====================================="
echo "🎉 Setup Complete!"
echo "====================================="
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run the Streamlit app:"
echo "     streamlit run app.py"
echo ""
echo "The app will open in your default browser at http://localhost:8501"
echo ""
