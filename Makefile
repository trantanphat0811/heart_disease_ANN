.PHONY: help install test lint format train eda report dashboard api web logs monitor clean docker-build docker-up docker-down

help:
	@echo "Heart Disease ML System - Development Commands"
	@echo "=============================================="
	@echo "Setup:"
	@echo "  make install          - Install dependencies"
	@echo "  make venv             - Create virtual environment"
	@echo ""
	@echo "Development:"
	@echo "  make train            - Train ML model"
	@echo "  make eda              - Run exploratory data analysis"
	@echo "  make report           - Generate HTML report"
	@echo "  make test             - Run unit tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make lint             - Run code linting"
	@echo "  make format           - Format code with black"
	@echo ""
	@echo "Running Services:"
	@echo "  make api              - Start FastAPI server"
	@echo "  make dashboard        - Start Streamlit dashboard"
	@echo "  make web              - Start web server (port 8000)"
	@echo "  make all-services     - Start all services"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-up        - Start containers"
	@echo "  make docker-down      - Stop containers"
	@echo "  make docker-logs      - View container logs"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs             - View application logs"
	@echo "  make monitor          - Start model monitoring"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove generated files"
	@echo "  make clean-docker     - Stop and remove containers"

venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8 bandit

train:
	@echo "🚀 Training model..."
	python3 src/train.py --data data/synthetic_heart_disease_dataset.csv
	@echo "✓ Model training complete"

eda:
	@echo "📊 Running exploratory data analysis..."
	python3 src/eda.py
	@echo "✓ EDA complete"

report: eda
	@echo "📄 Generating HTML report..."
	python3 scripts/generate_report_html.py
	@echo "✓ Report generated: web/index.html"

test:
	@echo "🧪 Running unit tests..."
	pytest tests/ -v
	@echo "✓ Tests complete"

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated: htmlcov/index.html"

lint:
	@echo "🔍 Linting code..."
	flake8 src/ scripts/ --max-line-length=100
	@echo "✓ Linting complete"

format:
	@echo "✨ Formatting code with black..."
	black src/ scripts/ tests/ --line-length=100
	isort src/ scripts/ tests/
	@echo "✓ Code formatted"

api:
	@echo "🚀 Starting FastAPI server..."
	python3 src/app.py

dashboard:
	@echo "📊 Starting Streamlit dashboard..."
	streamlit run src/dashboard_plotly.py

web:
	@echo "🌐 Starting web server on http://localhost:8000..."
	python3 -m http.server 8000 --directory web

all-services:
	@echo "🚀 Starting all services..."
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "Web: http://localhost:8080"
	@bash -c 'python3 src/app.py & streamlit run src/dashboard_plotly.py & python3 -m http.server 8000 --directory web'

logs:
	@echo "📋 Recent log entries:"
	@tail -50 logs/app_*.log 2>/dev/null || echo "No logs found"

monitor:
	@echo "📡 Starting model monitoring..."
	@python3 -c "from src.monitor import ModelMonitor; m = ModelMonitor(); print('✓ Monitor initialized')"

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t heart-disease-api:latest .
	@echo "✓ Docker image built"

docker-up:
	@echo "🚀 Starting Docker containers..."
	docker-compose up -d
	@echo "✓ Containers started"
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "Web: http://localhost:8080"

docker-down:
	@echo "🛑 Stopping Docker containers..."
	docker-compose down
	@echo "✓ Containers stopped"

docker-logs:
	docker-compose logs -f

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf logs/*.log
	@echo "✓ Cleanup complete"

clean-docker: docker-down
	@echo "🐳 Removing Docker images..."
	docker rmi heart-disease-api:latest 2>/dev/null || true
	@echo "✓ Docker cleanup complete"

.DEFAULT_GOAL := help
