#!/bin/bash
# 跑完用 Server酱 推送微信通知。SEND_KEY 从 .env 读取，不写进脚本（.env 已被 .gitignore）。
#
# 用法：把要跑的命令整条传给它即可——
#   ./notify.sh python scripts/run_gp_walkforward.py
#   nohup ./notify.sh python scripts/run_gp_walkforward.py &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 载入密钥（.env 不进版本库；CI/服务器也可直接 export SEND_KEY）=====
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

if [ -z "${SEND_KEY:-}" ]; then
    echo "缺少 SEND_KEY：请在 .env 配置，或先 export SEND_KEY=SCT..." >&2
    exit 1
fi

send_notify() {
    curl -s -X POST "https://sctapi.ftqq.com/${SEND_KEY}.send" \
        --data-urlencode "title=$1" \
        --data-urlencode "desp=$2" >/dev/null
}

# ===== 运行传入的命令，实时回显的同时留一份输出做摘要 =====
START_TS=$(date +%s)
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")

LOG=$(mktemp)
"$@" 2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}   # 取 "$@" 的退出码，而非 tee 的

END_TS=$(date +%s)
FINISH_TIME=$(date "+%Y-%m-%d %H:%M:%S")
ELAPSED=$((END_TS - START_TS))
DURATION=$(printf '%dh%02dm%02ds' $((ELAPSED / 3600)) $((ELAPSED % 3600 / 60)) $((ELAPSED % 60)))
OUTPUT=$(tail -20 "$LOG")
HOST=$(hostname)

if [ "$EXIT_CODE" -eq 0 ]; then
    TITLE="✅ 任务完成 · ${DURATION}"
else
    TITLE="❌ 任务失败（退出码 $EXIT_CODE） · ${DURATION}"
fi

send_notify "$TITLE" "**主机**: ${HOST}
**命令**: \`$*\`
**开始**: ${START_TIME}
**结束**: ${FINISH_TIME}
**耗时**: ${DURATION}
**退出码**: ${EXIT_CODE}

最后 20 行输出：
\`\`\`
${OUTPUT}
\`\`\`"

rm -f "$LOG"
exit "$EXIT_CODE"
