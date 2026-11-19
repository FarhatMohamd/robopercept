.PHONY: install dev test lint format demo demo-robot eval bench media docker

install:
	pip install -e .

dev:
	pip install -e ".[dev,onnx]"
	pre-commit install || true

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

demo:
	robopercept demo --scenario urban

demo-robot:
	robopercept demo --scenario warehouse

eval:
	robopercept eval --scenario urban --frames 900
	robopercept eval --scenario warehouse --frames 780

bench:
	robopercept bench -c urban --frames 600

media:
	python scripts/make_media.py --out docs/media

docker:
	docker compose up --build demo
