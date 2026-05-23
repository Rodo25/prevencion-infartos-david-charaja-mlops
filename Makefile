# Makefile — Pipeline CI/CD Local + Pre-producción
# Uso: make <target>

.PHONY: all lint test train validate docker preprod-up preprod-down preprod-logs smoke clean help

COMPOSE_FILE = docker-compose.preprod.yml
VERSION ?= latest

all: lint test train validate docker
	@echo "✓ Pipeline CI/CD local completado. Listo para git push."

lint:
	flake8 src/ tests/ api/ --config=setup.cfg --max-line-length=150 --ignore=E302,W291,F401,E402

test:
	pytest tests/ --ignore=tests/smoke -v --tb=short --cov=src --cov-report=term-missing

train:
	python src/01_prepare_data.py
	python src/02_train_model.py

validate:
	python src/03_validate_model.py

docker:
	docker build -t prevencion-infartos-api:local .

preprod-up:
	docker compose -f $(COMPOSE_FILE) up --build -d
	@echo "Esperando servicios..."
	sleep 45
	@echo "API: http://localhost:8000 | Docs: http://localhost:8000/docs | MLflow: http://localhost:5000"

preprod-down:
	docker compose -f $(COMPOSE_FILE) down -v

preprod-logs:
	docker compose -f $(COMPOSE_FILE) logs -f

prod-build:
	docker build -t prevencion-infartos-api:1.0.0 .

prod-up:
	docker run -d --name prevencion-infartos-api-prod -p 8000:8000 prevencion-infartos-api:1.0.0

prod-down:
	docker stop prevencion-infartos-api-prod || true
	docker rm prevencion-infartos-api-prod || true

smoke:
	pytest tests/smoke/ -v --tb=short

clean:
	rm -rf artifacts/*.pkl artifacts/*.json artifacts/*.txt artifacts/*.png mlruns/ .coverage htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Limpieza completada."

help:
	@echo "make all          — lint + test + train + validate + docker"
	@echo "make train        — prepara datos y entrena modelo con MLflow"
	@echo "make validate     — quality gate"
	@echo "make preprod-up   — levanta API + trainer + MLflow"
	@echo "make smoke        — smoke tests del stack"
