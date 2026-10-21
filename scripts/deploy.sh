#!/usr/bin/env bash
# Runs ON the server (piped by `make deploy` via ssh): deploy.sh <app-dir-under-~/apps> <image:tag>
# Pulls first, writes the image tag into .env (compose.yaml references it as ${APP_IMAGE} -- no sed
# on compose.yaml itself, so there's no risk of a pattern match hitting a dependency service's own
# image line), then up -d, then waits for /health.
# The body is one { } block so bash reads all of it before running: `docker compose exec` reads
# stdin, and would otherwise eat the rest of this piped script (same gotcha as oxbot-miniapp's deploy.sh).
{
set -euo pipefail
dir="$1" image="$2"
cd "$HOME/apps/$dir"

docker pull -q "$image"
touch .env
grep -v '^APP_IMAGE=' .env > .env.tmp || true
echo "APP_IMAGE=$image" >> .env.tmp
mv .env.tmp .env
docker compose up -d

for i in $(seq 30); do
  # Python app (not bun like hello): the API serves /health on port 8000
  docker compose exec -T app python -c 'import urllib.request; urllib.request.urlopen("http://localhost:8000/health")' 2>/dev/null && break
  [ "$i" = 30 ] && { echo "no healthy /health after 30s" >&2; docker compose logs --tail 20 app; exit 1; }
  sleep 1
done
docker compose logs --tail 5 app
exit
}
