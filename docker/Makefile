# AutocleanEEG Serve - Docker Management
# Usage: make [target]
#
# Quick start:
#   1. Copy your serve-config.yaml to ./config/
#   2. Run: make build && make start
#   3. Access API at http://localhost:4000

.PHONY: help start stop restart logs status shell health queue clean build push

# Default target - show help
help:
	@echo "AutocleanEEG Serve Commands:"
	@echo ""
	@echo "  make build     - Build the container"
	@echo "  make start     - Start the container"
	@echo "  make stop      - Stop the container"
	@echo "  make restart   - Restart (picks up config changes)"
	@echo "  make logs      - Follow container logs"
	@echo "  make status    - Show container and queue status"
	@echo "  make shell     - Open bash shell in container"
	@echo "  make health    - Check system health"
	@echo "  make queue     - Show queue statistics"
	@echo "  make clean     - Stop and remove container + volumes"
	@echo "  make push      - Push to container registry"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit config/serve-config.yaml, then run 'make restart'"

# Build container
build:
	docker-compose build
	@echo "Container built successfully"

# Start container
start:
	docker-compose up -d
	@echo "Container started. API at http://localhost:4000"
	@echo "Docs at http://localhost:4000/docs"

# Stop container
stop:
	docker-compose down

# Restart (use after config changes)
restart:
	docker-compose restart
	@echo "Container restarted with updated configuration"

# Follow logs
logs:
	docker-compose logs -f

# Show status
status:
	@echo "=== Container Status ==="
	@docker-compose ps
	@echo ""
	@echo "=== Queue Status ==="
	@curl -s http://localhost:4000/api/queue/stats 2>/dev/null | python3 -m json.tool || echo "API not responding"

# Open shell
shell:
	docker exec -it autoclean bash

# Health check
health:
	@curl -s http://localhost:4000/health | python3 -m json.tool

# Queue stats
queue:
	@curl -s http://localhost:4000/api/queue/stats | python3 -m json.tool

# Clean up everything
clean:
	docker-compose down -v
	@echo "Container and volumes removed"

# Push to registry (configure REGISTRY variable or use default)
REGISTRY ?= ghcr.io/cincibrainlab
VERSION ?= latest

push: build
	docker tag autoclean/serve:latest $(REGISTRY)/autoclean-serve:$(VERSION)
	docker push $(REGISTRY)/autoclean-serve:$(VERSION)
	@echo "Pushed to $(REGISTRY)/autoclean-serve:$(VERSION)"
