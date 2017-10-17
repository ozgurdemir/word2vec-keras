.PHONY: run test example clean download

TRAIN_SET_URL="http://mattmahoney.net/dc/text8.zip"

DIR=$(shell pwd)
DATA_DIR=$(DIR)/data
TEXT_DATA=$(DATA_DIR)/text8
ZIPPED_TEXT_DATA="$(TEXT_DATA).zip"

run:
	docker run -it -t --rm -v $(DIR):/srv/ai gw000/keras /bin/bash

test:
	echo $(DATA_DIR)

example:
	docker run -d --rm --name word2vec-keras -v $(DIR):/srv/ai gw000/keras python ai/src/main.py --train ai/data/text8

download: $(TEXT_DATA)

$(TEXT_DATA): $(ZIPPED_TEXT_DATA)
	unzip -o $(ZIPPED_TEXT_DATA) -d $(DATA_DIR)
	touch $@

$(ZIPPED_TEXT_DATA):
	mkdir -p $(DATA_DIR)
	wget $(TRAIN_SET_URL) -O $@
	touch $@

clean:
	rm data/*