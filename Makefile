.PHONY: run-it test train predict clean

TRAIN_SET_URL="http://mattmahoney.net/dc/text8.zip"

DIR=$(shell pwd)
DATA_DIR=$(DIR)/data
DOCKER_IMAGE="gw000/keras:2.1.1-py3-tf-cpu"
CONTAINER_NAME="word2vec-keras"
DOCKER_RUN=docker run --rm --name $(CONTAINER_NAME) -v $(DIR):/srv/ai -w /srv/ai

run-it:
	$(DOCKER_RUN) -it $(DOCKER_IMAGE) /bin/bash

test:
	$(DOCKER_RUN) $(DOCKER_IMAGE) python3 -m unittest discover src

train: data/text8
	$(DOCKER_RUN) $(DOCKER_IMAGE) python3 src/train.py --train data/text8 --embeddings data/

predict:
	$(DOCKER_RUN) -it $(DOCKER_IMAGE) python3 src/predict.py --embeddings data/

data/text8: data/text8.zip
	unzip -o data/text8.zip -d $(DATA_DIR)
	touch $@

data/text8.zip:
	mkdir -p $(DATA_DIR)
	wget $(TRAIN_SET_URL) -O $@
	touch $@

clean:
	rm data/*