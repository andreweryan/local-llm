#!/usr/bin/env bash
# chat.sh — terminal chat client for local-llm API

SESSION="${1:-$(python3 -c 'import uuid; print(uuid.uuid4())')}"
DOCS_FOLDER="${2:-10}"
TOP_K="${3:-10}"
API_URL="http://localhost:8000"  # always local
export SESSION_ID="$SESSION"
export DOCS_FOLDER="$DOCS_FOLDER"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE=$SCRIPT_DIR/logs/server.log
SERVER_PID=""
TAIL_PID=""

BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'


command -v curl >/dev/null || { echo "curl required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

health_ready() {
    curl -sf "$API_URL/health" | grep -q '"ready": *true'
}

# ---------------------------------------------------------------------------
# Start server if needed
# ---------------------------------------------------------------------------

if ! health_ready; then
    echo -e "${DIM}Server not running — starting main.py...${RESET}"

    python3 -u "$SCRIPT_DIR/main.py" >"$LOG_FILE" 2>&1 &

    SERVER_PID=$!

    echo -e "${DIM}Server PID: $SERVER_PID  |  logs: $LOG_FILE${RESET}"
    echo -e "${DIM}Waiting for server to be ready...${RESET}"

    ready=0
    for _ in $(seq 1 600); do  # up to 10 minutes
        sleep 1
        if health_ready; then
            ready=1
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo -e "\n${RED}Server exited — check $LOG_FILE${RESET}"
            kill "$TAIL_PID" 2>/dev/null
            exit 1
        fi
    done

    kill "$TAIL_PID" 2>/dev/null
    wait "$TAIL_PID" 2>/dev/null

    # echo -e "${DIM}Server ready.${RESET}"
else
    echo -e "${DIM}Server already running at $API_URL${RESET}"
fi

# ---------------------------------------------------------------------------
# Shutdown handling
# ---------------------------------------------------------------------------

cleanup() {
    trap - EXIT INT TERM
    echo ""
    if [[ -n "$SERVER_PID" ]]; then
        echo -e "${DIM}Stopping server (PID $SERVER_PID)...${RESET}"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    echo -e "${DIM}Goodbye.${RESET}"
    exit 0
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

echo -e "${DIM}session name: $SESSION${RESET}"
echo -e "${DIM}Commands: exit | sources | session${RESET}"
echo -e "${DIM}────────────────────────────────────────────────────${RESET}"

LAST_SOURCES=""

while true; do
    echo -en "${CYAN}You: ${RESET}"
    IFS= read -r prompt
    [[ -z "$prompt" ]] && continue

    case "$prompt" in
        exit|quit|q) break ;;
        sources)
            if [[ -z "$LAST_SOURCES" || "$LAST_SOURCES" == "[]" ]]; then
                echo -e "${DIM}No sources from last query.${RESET}"
            else
                echo -e "${YELLOW}Sources:${RESET}"
                python3 << 'EOF'
import sys, json, os
sources = json.loads(os.environ["LAST_SOURCES"])
for i, s in enumerate(sources, 1):
    page = f" p.{s.get('page')}" if s.get("page") else ""
    score = s.get("rerank_score", "")
    print(f"  {i}. {s['source']}{page}  [score: {score}]")
EOF
            fi
            continue
            ;;
        session)
            echo -e "${DIM}Session ID: $SESSION${RESET}"
            continue
            ;;
    esac

    JSON_PROMPT=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$prompt")
    PAYLOAD="{\"prompt\": $JSON_PROMPT, \"top_k\": $TOP_K, \"session_id\": \"$SESSION\"}"

    # Retry loop for initial warm-up
    RESPONSE=""
    for _ in $(seq 1 3); do
        RESPONSE=$(curl -s --max-time 120 -X POST "$API_URL/generate" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD")
        [[ -n "$RESPONSE" ]] && break
        sleep 1
    done

    if [[ -z "$RESPONSE" ]]; then
        echo -e "${RED}No response from API — server might still be loading.${RESET}"
        continue
    fi

    ANSWER=$(python3 -c '
import sys,json
try:
    d=json.loads(sys.argv[1])
    print(d.get("response",""))
except Exception as e:
    print(f"Parse error: {e}")
' "$RESPONSE")

    REWRITTEN=$(python3 -c 'import sys,json; print(json.loads(sys.argv[1]).get("rewritten_query",""))' "$RESPONSE" 2>/dev/null)
    LAST_SOURCES=$(python3 -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]).get("sources",[])))' "$RESPONSE" 2>/dev/null)
    export LAST_SOURCES
    SOURCE_COUNT=$(python3 -c 'import sys,json; print(len(json.loads(sys.argv[1])))' "$LAST_SOURCES" 2>/dev/null)

    echo -e "${GREEN}Assistant:${RESET} $ANSWER"

    META=""
    [[ -n "$REWRITTEN" && "$REWRITTEN" != "$prompt" ]] && META="${DIM}↳ query rewritten${RESET}"
    [[ "$SOURCE_COUNT" -gt 0 ]] && META="${META}  ${DIM}[$SOURCE_COUNT source(s) — type 'sources']${RESET}"
    [[ -n "$META" ]] && echo -e "$META"
    echo
done
