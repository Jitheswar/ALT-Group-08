#!/usr/bin/env bash
# Sets up and starts the ATL resume screening web demo.
#
# Usage:
#   ./start.sh              # scripted model client (no network, no API key)
#   ./start.sh --live       # real deepseek-v4-flash provider, needs DEEPSEEK_API_KEY
#   PORT=8080 ./start.sh    # override the port (default 5000)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required but not found on PATH." >&2
    echo "Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

LIVE=0
if [[ "${1:-}" == "--live" ]]; then
    LIVE=1
fi

if [[ "$LIVE" == "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "error: --live requires DEEPSEEK_API_KEY to be set." >&2
    exit 1
fi

echo "==> Syncing dependencies (uv sync)"
uv sync

export PORT="${PORT:-5000}"

if [[ "$LIVE" == "1" ]]; then
    export SCREENING_WEB_LIVE=1
    echo "==> Starting screening-web on port $PORT (live: deepseek-v4-flash)"
else
    unset SCREENING_WEB_LIVE
    echo "==> Starting screening-web on port $PORT (scripted model client, no network)"
fi

exec uv run screening-web
