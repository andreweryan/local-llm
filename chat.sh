#!/bin/bash
# chat.sh — terminal chat client for local-llm API
# Usage: bash chat.sh [session_id] [top_k] [api_url]

API_URL="${3:-http://localhost:8000}"
TOP_K="${2:-10}"
SESSION="${1:-$(python3 -c 'import uuid; print(uuid.uuid4())')}"

# Colors
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required"; exit 1; }

echo -e "${BOLD}local-llm chat${RESET}  ${DIM}session: $SESSION${RESET}"
echo -e "${DIM}Commands: 'exit' to quit | 'sources' to show last sources | 'clear' to reset memory | 'session' to print session id${RESET}"
echo -e "${DIM}────────────────────────────────────────────────────${RESET}"

LAST_SOURCES=""

while true; do
    echo -en "${CYAN}You: ${RESET}"
    read -r prompt

    [[ -z "$prompt" ]] && continue

    case "$prompt" in
        exit|quit|q)
            echo -e "${DIM}Goodbye.${RESET}"
            break
            ;;
        sources)
            if [[ -z "$LAST_SOURCES" ]]; then
                echo -e "${DIM}No sources from last query.${RESET}"
            else
                echo -e "${YELLOW}Sources:${RESET}"
                echo "$LAST_SOURCES" | python3 -c "
import sys, json
sources = json.load(sys.stdin)
for i, s in enumerate(sources, 1):
    page = f\" p.{s['page']}\" if s.get('page') else ''
    score = s.get('rerank_score', '')
    print(f\"  {i}. {s['source']}{page}  [score: {score}]\")
" 2>/dev/null || echo "$LAST_SOURCES"
            fi
            continue
            ;;
        clear)
            curl -s -X DELETE "$API_URL/memory/$SESSION" >/dev/null
            echo -e "${DIM}Memory cleared.${RESET}"
            continue
            ;;
        session)
            echo -e "${DIM}Session ID: $SESSION${RESET}"
            continue
            ;;
    esac

    # Escape prompt for JSON
    JSON_PROMPT=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$prompt")

    RESPONSE=$(curl -s -X POST "$API_URL/generate" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": $JSON_PROMPT, \"top_k\": $TOP_K, \"session_id\": \"$SESSION\"}")

    if [[ $? -ne 0 ]] || [[ -z "$RESPONSE" ]]; then
        echo -e "${RED}Error: Could not reach API at $API_URL${RESET}"
        continue
    fi

    ERROR=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('detail',''))" 2>/dev/null)
    if [[ -n "$ERROR" ]]; then
        echo -e "${RED}API Error: $ERROR${RESET}"
        continue
    fi

    ANSWER=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response',''))" 2>/dev/null)
    REWRITTEN=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); q=d.get('rewritten_query',''); print(q) if q else None" 2>/dev/null)
    LAST_SOURCES=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('sources',[])))" 2>/dev/null)
    SOURCE_COUNT=$(echo "$LAST_SOURCES" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)

    echo -e "${GREEN}Assistant:${RESET} $ANSWER"

    # Show metadata line if rewrite happened or sources exist
    META=""
    if [[ -n "$REWRITTEN" && "$REWRITTEN" != "$prompt" && "$REWRITTEN" != "None" ]]; then
        META="${DIM}↳ query rewritten${RESET}"
    fi
    if [[ -n "$SOURCE_COUNT" && "$SOURCE_COUNT" -gt 0 ]]; then
        [[ -n "$META" ]] && META="$META  "
        META="${META}${DIM}[$SOURCE_COUNT source(s) — type 'sources' to list]${RESET}"
    fi
    [[ -n "$META" ]] && echo -e "$META"
    echo
done
