#!/bin/bash
# ML Studio Quick Start Script

echo "🔬 ML Studio - Universal ML Platform"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo -e "${GREEN}✓ Python found$(python3 --version)${NC}"
echo ""

# Check Node.js
echo -e "${BLUE}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found ($(node --version))${NC}"
echo ""

# Backend setup
echo -e "${BLUE}Setting up backend...${NC}"
cd backend
pip install -r requirements.txt
cd ..
echo -e "${GREEN}✓ Backend dependencies installed${NC}"
echo ""

# Frontend setup
echo -e "${BLUE}Setting up frontend...${NC}"
cd frontend
npm install
cd ..
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
echo ""

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "Terminal 1 (MLflow tracking):"
echo "  mlflow server --host 0.0.0.0 --port 5000"
echo ""
echo "Terminal 2 (Backend API):"
echo "  cd backend && uvicorn main:app --reload --port 8000"
echo ""
echo "Terminal 3 (Frontend):"
echo "  cd frontend && npm run dev"
echo ""
echo "Then open:"
echo "  🎨 Frontend: http://localhost:5173"
echo "  📈 MLflow UI: http://localhost:5000"
echo ""
