.PHONY: build run

DIR=$(shell pwd)

APP_NAME=word2vec-keras

build: ## Build the container
	docker build -t $(APP_NAME) .

build-nc: ## Build the container without caching
	docker build --no-cache -t $(APP_NAME) .

run-it: ## Run container interactive configured in `config.env`
	docker run -i -t --rm --name="$(APP_NAME)" $(APP_NAME) /bin/bash

run:
	docker run -it -t --rm -v $(DIR):/srv/ai gw000/keras /bin/bash