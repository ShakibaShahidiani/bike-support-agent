# Copy this into each app directory. For a plain app, just change APP and NAME to match (same value).
# For a staging instance of an existing app, change only APP (the server-side directory name, e.g.
# hello-staging) -- leave NAME as the real app's name (e.g. hello) so staging and prod share one image,
# never two separately-built ones.
APP   := bike
NAME  := bike
SHA   := $(shell git rev-parse --short HEAD)
IMAGE := docker.tail2ee28.ts.net/shakiba/$(NAME):$(SHA)
VM    := ubuntu@shakibas-vm

.PHONY: build deploy clean

# Refuse to build from uncommitted changes -- the SHA tag must mean something.
clean:
	@git diff --quiet HEAD || { echo "uncommitted changes: commit first"; exit 1; }

build: clean
	docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
		-t $(IMAGE) -t docker.tail2ee28.ts.net/shakiba/$(NAME):latest --push .

# scripts/deploy.sh runs ON the server: pulls, writes the exact SHA tag into .env as APP_IMAGE=
# (compose.yaml reads it via ${APP_IMAGE} -- never edits compose.yaml itself), up -d, waits for
# /health. Keep that script's shape if you copy this -- it assumes a service named exactly `app`
# with a `/health` route (docker compose exec -T app ...).
deploy: build
	rsync -az compose.yaml $(VM):~/apps/$(APP)/compose.yaml
	ssh $(VM) bash -s -- $(APP) $(IMAGE) < scripts/deploy.sh
