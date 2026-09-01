#!/usr/bin/env bash
# Deploy this app to Azure App Service (Linux, Python).
#
# Encodes what cost three failed starts the first time. Read the comments before
# changing any of it -- each one is a failure that looked like something else.
#
#   ./deploy/azure.sh <resource-group> <app-name> [location]
set -euo pipefail

RG="${1:?resource group}"; APP="${2:?app name}"; LOC="${3:-westus2}"
PLAN="${APP,,}-b1-plan"
STAGE="$(mktemp -d)"; ZIP="$STAGE.zip"
trap 'rm -rf "$STAGE" "$ZIP"' EXIT

say() { printf '\n== %s\n' "$*"; }

say "plan and app"
az appservice plan create -g "$RG" -n "$PLAN" --is-linux --sku B1 -l "$LOC" -o none
az webapp create -g "$RG" -p "$PLAN" -n "$APP" --runtime "PYTHON:3.13" -o none

say "staging"
rsync -a --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
      --exclude '.pytest_cache' --exclude '.ruff_cache' --exclude '*.egg-info' \
      --exclude '.env' --exclude 'benchmarks' --exclude 'tests' \
      --exclude 'data/.cache' --exclude '.query-embeddings.json' \
      ./ "$STAGE/"

# Oryx installs from requirements.txt and ignores pyproject.toml. Without a
# trailing "." pip never installs the project, so the console script does not
# exist and the container dies with exit code 127.
sed -n '/^dependencies/,/]/p' pyproject.toml | grep '"' | sed 's/.*"\(.*\)".*/\1/' > "$STAGE/requirements.txt"
echo "." >> "$STAGE/requirements.txt"

# The embedding cache is large and must survive restarts, so it lives in
# /home/data (persistent) rather than the package or the instance disk
# (ephemeral -- re-embedding on every restart costs real money).
say "settings"
az webapp config appsettings set -g "$RG" -n "$APP" --settings \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true \
  WEBSITES_CONTAINER_START_TIME_LIMIT=1800 \
  ASKHUB_HOST=0.0.0.0 ASKHUB_PORT=8000 \
  ASKHUB_CACHE_DIR=/home/data/embedcache \
  ASKHUB_MCP_ALLOWED_HOSTS="${APP,,}.azurewebsites.net" \
  OPENROUTER_APP_URL="https://${APP,,}.azurewebsites.net" \
  OPENROUTER_APP_TITLE="$APP" \
  -o none
# Secrets come from the local .env and are never echoed.
if [ -f .env ]; then
  KEY=$(grep -E '^OPENROUTER_API_KEY=' .env | cut -d= -f2- | tr -d '"'"'"' \r')
  [ -n "$KEY" ] && az webapp config appsettings set -g "$RG" -n "$APP" \
    --settings OPENROUTER_API_KEY="$KEY" -o none
fi

# Relative paths only: with Oryx the app runs from an extracted temp directory,
# not /home/site/wwwroot, so absolute paths into wwwroot resolve to nothing.
az webapp config set -g "$RG" -n "$APP" --always-on true \
  --startup-file "ask-hf-hub --host 0.0.0.0 --port 8000" -o none

# Deploying to a stopped site silently does nothing while still exiting 0.
say "ensuring the site is running before deploying"
az webapp start -g "$RG" -n "$APP" -o none || true
until [ "$(az webapp show -g "$RG" -n "$APP" --query state -o tsv)" = "Running" ]; do sleep 5; done

say "deploying"
(cd "$STAGE" && zip -qr "$ZIP" .)
az webapp deploy -g "$RG" -n "$APP" --src-path "$ZIP" --type zip --timeout 1800 >/dev/null

# az exiting 0 and Kudu reporting success both lie. Read a file back.
say "verifying the package actually landed"
read -r U P < <(az webapp deployment list-publishing-credentials -g "$RG" -n "$APP" \
  --query "[publishingUserName,publishingPassword]" -o tsv)
REMOTE=$(curl -fsS -u "$U:$P" "https://${APP,,}.scm.azurewebsites.net/api/vfs/site/wwwroot/requirements.txt" | tail -1)
if [ "$REMOTE" != "." ]; then
  echo "FAILED: requirements.txt on the server does not end with '.' -- the package did not land." >&2
  echo "Check the site is Running and redeploy; do not trust the exit code." >&2
  exit 1
fi
echo "requirements.txt verified on server"

# The recycle a deploy triggers fires against the *previous* build output, so
# the site comes back healthy still serving the old assets. Observed twice: the
# package is verifiably on the server, health returns 200, and the browser gets
# the previous version. An explicit restart after the build is what picks it up.
say "restarting onto the new build"
az webapp restart -g "$RG" -n "$APP" -o none
sleep 30

say "waiting for health"
for _ in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "https://${APP,,}.azurewebsites.net/health" || true)
  [ "$code" = "200" ] && { echo "healthy"; break; }
  echo "  HTTP ${code:-000}"; sleep 15
done

# Health alone does not prove the new code is being served -- check the asset
# version actually changed, and restart once more if it has not.
WANT=$(grep -o 'v=[0-9]*' src/askhub/static/index.html | head -1)
GOT=$(curl -s "https://${APP,,}.azurewebsites.net/" | grep -o 'v=[0-9]*' | head -1)
if [ -n "$WANT" ] && [ "$WANT" != "$GOT" ]; then
  echo "serving $GOT, expected $WANT -- restarting again"
  az webapp restart -g "$RG" -n "$APP" -o none
  sleep 45
  until curl -s --max-time 12 "https://${APP,,}.azurewebsites.net/health" | grep -q '"status"'; do sleep 15; done
  GOT=$(curl -s "https://${APP,,}.azurewebsites.net/" | grep -o 'v=[0-9]*' | head -1)
fi
echo "serving assets: ${GOT:-unknown} (expected ${WANT:-n/a})"
curl -s "https://${APP,,}.azurewebsites.net/health" | python3 -m json.tool | head -8
cat <<EOF

Deployed: https://${APP,,}.azurewebsites.net

If health reports "embedding_cache_hit": false, the persistent cache is missing.
Upload it once -- it survives restarts, the instance disk does not:

  (cd data/.cache && zip -qr /tmp/cache.zip *.npz)
  curl -u "\$U:\$P" -X PUT --data-binary @/tmp/cache.zip \\
    "https://${APP,,}.scm.azurewebsites.net/api/zip/data/embedcache/"
EOF
