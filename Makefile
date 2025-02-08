IMAGE_NAME = grpo_unsloth
CONTAINER_NAME = grpo_unsloth_container

.PHONY: build create start stop clean

build:
	docker build -t $(IMAGE_NAME) .

create:
	docker create -it \
		--gpus=all \
		--name $(CONTAINER_NAME) \
		-v $$(pwd)/models:/models \
		-v $$(pwd):/workspace \
		-e HF_HOME=/models/cache \
		$(IMAGE_NAME)

start:
	docker start $(CONTAINER_NAME)

dry_run:
	docker exec -it $(CONTAINER_NAME) bash -c "uv run python main.py 'saving=null' 'training.max_steps=1000'"

train:
	docker exec -it $(CONTAINER_NAME) bash -c "uv run python main.py"

stop:
	docker stop $(CONTAINER_NAME)

clean:
	docker rm $(CONTAINER_NAME)

# Combined targets
up: build create start dry_run

down: stop clean
