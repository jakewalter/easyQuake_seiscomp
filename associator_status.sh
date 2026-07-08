#!/usr/bin/env bash
# associator_status.sh – report status and pick statistics for all three associators

ASSOC_DB="/tmp/scphasepapy/assoc.db"
AUTOLOC_LOG="$HOME/.seiscomp/log/scautoloc.log"
PHASEPAPY_LOG="$HOME/.seiscomp/log/scphasepapy.log"
PYOCTO_LOG="$HOME/.seiscomp/log/scpyocto.log"
NOW_UTC=$(date -u '+%Y-%m-%d %H:%M:%S')

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

module_status() {
    local mod="$1"
    if seiscomp status "$mod" 2>/dev/null | grep -q 'is running'; then
        echo -e "${GREEN}RUNNING${RESET}"
    else
        echo -e "${RED}NOT RUNNING${RESET}"
    fi
}

last_log_activity() {
    local logfile="$1"
    if [[ -f "$logfile" ]]; then
        local last
        last=$(grep -oP '^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}' "$logfile" | tail -1)
        echo "${last:-n/a}"
    else
        echo "no log"
    fi
}

last_log_errors() {
    local logfile="$1"
    local since_min="${2:-60}"
    if [[ -f "$logfile" ]]; then
        local cutoff
        cutoff=$(date -u -d "${since_min} minutes ago" '+%Y/%m/%d %H:%M:%S' 2>/dev/null || \
                 date -u -v-${since_min}M '+%Y/%m/%d %H:%M:%S' 2>/dev/null)
        grep '\[error\]' "$logfile" | \
            awk -v c="$cutoff" '$1" "$2 >= c' | \
            grep -v 'latency' | wc -l
    else
        echo "0"
    fi
}

latency_errors() {
    local logfile="$1"
    local since_min="${2:-60}"
    if [[ -f "$logfile" ]]; then
        local cutoff
        cutoff=$(date -u -d "${since_min} minutes ago" '+%Y/%m/%d %H:%M:%S' 2>/dev/null || \
                 date -u -v-${since_min}M '+%Y/%m/%d %H:%M:%S' 2>/dev/null)
        grep 'latency level' "$logfile" | \
            awk -v c="$cutoff" '$1" "$2 >= c' | wc -l
    else
        echo "0"
    fi
}

last_heartbeat() {
    local logfile="$1"
    local pattern="$2"
    if [[ -f "$logfile" ]]; then
        local line
        line=$(grep "$pattern" "$logfile" | tail -1)
        if [[ -n "$line" ]]; then
            echo "$line" | grep -oP '^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}'
        else
            echo "none yet"
        fi
    else
        echo "no log"
    fi
}


TOTAL_RAM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
SC_RUN_DIR="${SEISCOMP_ROOT:-$HOME/seiscomp}/var/run"

mem_info() {
    # Prints "RSS  VSZ  Swap  Threads" for a module, or dashes if not running
    local mod="$1"
    local pid
    pid=$(cat "${SC_RUN_DIR}/${mod}.pid" 2>/dev/null)
    if [[ -n "$pid" ]] && [[ -d /proc/$pid ]]; then
        local rss vsz swap thr pct
        rss=$(awk '/VmRSS/{print $2}'   /proc/$pid/status 2>/dev/null)
        vsz=$(awk '/VmSize/{print $2}'  /proc/$pid/status 2>/dev/null)
        swap=$(awk '/VmSwap/{print $2}' /proc/$pid/status 2>/dev/null)
        thr=$(awk '/Threads/{print $2}' /proc/$pid/status 2>/dev/null)
        pct=$(awk "BEGIN{printf \"%.1f\", ${rss:-0}/${TOTAL_RAM_KB}*100}")
        # Format RSS/VSZ in MB
        local rss_mb vsz_mb swap_mb
        rss_mb=$(( ${rss:-0} / 1024 ))
        vsz_mb=$(( ${vsz:-0} / 1024 ))
        swap_mb=$(( ${swap:-0} / 1024 ))
        echo "$pid $rss_mb $vsz_mb $swap_mb $pct $thr"
    else
        echo "- - - - - -"
    fi
}

color_mb() {
    # Colour-code RSS: green <500MB, yellow 500-2000MB, red >2000MB
    local mb="$1"
    if [[ "$mb" == "-" ]]; then echo -e "${GREEN}-${RESET}"; return; fi
    if   (( mb > 2000 )); then echo -e "${RED}${mb} MB${RESET}"
    elif (( mb > 500  )); then echo -e "${YELLOW}${mb} MB${RESET}"
    else                       echo -e "${GREEN}${mb} MB${RESET}"
    fi
}

color_swap() {
    # Colour-code Swap: green=0, yellow>0 (any swap usage is noteworthy)
    local mb="$1"
    if [[ "$mb" == "-" ]]; then echo -e "${GREEN}-${RESET}"; return; fi
    if (( mb > 0 )); then echo -e "${YELLOW}${mb} MB${RESET}"
    else                  echo -e "${GREEN}${mb} MB${RESET}"
    fi
}

echo -e "${BOLD}==========================================${RESET}"
echo -e "${BOLD}  Associator Status Report${RESET}"
echo -e "${BOLD}  UTC: ${NOW_UTC}${RESET}"
echo -e "${BOLD}==========================================${RESET}"
echo

# --- Module running status ---
echo -e "${CYAN}${BOLD}Module Status${RESET}"
printf "  %-18s %s\n" "scautoloc"   "$(module_status scautoloc)"
printf "  %-18s %s\n" "scphasepapy" "$(module_status scphasepapy)"
printf "  %-18s %s\n" "scpyocto"    "$(module_status scpyocto)"
echo

# --- RAM usage ---
TOTAL_RAM_MB=$(( TOTAL_RAM_KB / 1024 ))
echo -e "${CYAN}${BOLD}Memory Usage  (system total: ${TOTAL_RAM_MB} MB)${RESET}"
printf "  %-18s  %-6s  %-8s  %-8s  %-8s  %-6s  %s\n" \
    "Module" "PID" "RSS" "VSZ" "Swap" "%RAM" "Threads"
printf "  %-18s  %-6s  %-8s  %-8s  %-8s  %-6s  %s\n" \
    "------" "---" "---" "---" "----" "----" "-------"
for mod in scautoloc scphasepapy scpyocto; do
    read -r pid rss_mb vsz_mb swap_mb pct thr <<< "$(mem_info $mod)"
    printf "  %-18s  %-6s  %-17s  %-8s  %-17s  %-6s  %s\n" \
        "$mod" "$pid" "$(color_mb $rss_mb)" "${vsz_mb} MB" "$(color_swap $swap_mb)" "${pct}%" "$thr"
done
echo

# --- Last log activity + errors (last 60 min) ---
echo -e "${CYAN}${BOLD}Last Log Activity  (errors / latency warnings in last 60 min)${RESET}"
for mod in scautoloc scphasepapy scpyocto; do
    logfile="$HOME/.seiscomp/log/${mod}.log"
    last=$(last_log_activity "$logfile")
    errs=$(last_log_errors "$logfile" 60)
    lats=$(latency_errors "$logfile" 60)
    if [[ "$errs" -gt 0 ]]; then
        err_str="${RED}${errs} error(s)${RESET}"
    else
        err_str="${GREEN}0 errors${RESET}"
    fi
    if [[ "$lats" -gt 0 ]]; then
        lat_str="${YELLOW}${lats} latency${RESET}"
    else
        lat_str="${GREEN}0 latency${RESET}"
    fi
    printf "  %-18s last=%-22s  %s  %s\n" "$mod" "$last" "$err_str" "$lat_str"
done
echo

# --- Heartbeat check for our two custom associators ---
echo -e "${CYAN}${BOLD}Associator Heartbeats  (logged every 5 min; 'none yet' is OK after recent restart)${RESET}"
hb_pp=$(last_heartbeat "$HOME/.seiscomp/log/scphasepapy.log" "heartbeat")
hb_oc=$(last_heartbeat "$HOME/.seiscomp/log/scpyocto.log"    "heartbeat")
printf "  %-18s last heartbeat=%-22s\n" "scphasepapy" "$hb_pp"
printf "  %-18s last heartbeat=%-22s\n" "scpyocto"    "$hb_oc"
echo

# --- scphasepapy SQLite stats ---
echo -e "${CYAN}${BOLD}scphasepapy Pick Buffer  (${ASSOC_DB})${RESET}"
if [[ -f "$ASSOC_DB" ]]; then
    python3 << PYEOF
import sqlite3, datetime, sys

db = "$ASSOC_DB"
now = datetime.datetime.utcnow()

try:
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    cur.execute('SELECT count(*) FROM picks')
    n_picks = cur.fetchone()[0]

    cur.execute('SELECT max(time) FROM picks')
    latest_raw = cur.fetchone()[0]

    cur.execute('SELECT count(*) FROM picks_modified WHERE assoc_id IS NULL')
    n_pm = cur.fetchone()[0]

    cur.execute('SELECT count(*) FROM candidate WHERE assoc_id IS NULL')
    n_cand = cur.fetchone()[0]

    cur.execute('SELECT count(*) FROM associated')
    n_assoc = cur.fetchone()[0]

    # Staleness of latest pick
    if latest_raw:
        try:
            fmt = '%Y-%m-%d %H:%M:%S.%f' if '.' in latest_raw else '%Y-%m-%d %H:%M:%S'
            latest_dt = datetime.datetime.strptime(latest_raw, fmt)
            age_s = (now - latest_dt).total_seconds()
            age_str = f'{age_s:.0f}s ago'
            freshness = '\x1b[0;32mFRESH\x1b[0m' if age_s < 120 else '\x1b[0;31mSTALE\x1b[0m'
        except Exception:
            age_str = 'n/a'
            freshness = ''
    else:
        latest_raw = 'none'
        age_str = 'n/a'
        freshness = '\x1b[0;31mNO PICKS\x1b[0m'

    # pm:picks ratio health
    ratio = n_pm / n_picks if n_picks > 0 else 0
    ratio_flag = '\x1b[0;32mOK\x1b[0m' if ratio <= 1.5 else '\x1b[0;31mHIGH\x1b[0m'

    print(f'  Picks buffered       : {n_picks}')
    print(f'  Latest pick          : {latest_raw}  ({age_str})  {freshness}')
    print(f'  picks_modified       : {n_pm}  (ratio {ratio:.1f})  {ratio_flag}')
    print(f'  Candidates (unassoc) : {n_cand}')
    print(f'  Associated events    : {n_assoc}')

    # Recent picks (last 10)
    print()
    print('  Last 10 picks (sta, net, time, phase):')
    cur.execute('SELECT sta, net, time, phase FROM picks ORDER BY time DESC LIMIT 10')
    for row in cur.fetchall():
        print(f'    {row[1]}.{row[0]:<8s}  {row[2]}  {row[3] or "?"}')

    # S-P pairs within TT range
    cur.execute('SELECT sta, net, time FROM picks ORDER BY sta, net, time')
    rows = cur.fetchall()
    conn.close()

    pairs = []
    from itertools import groupby
    for key, grp in groupby(rows, key=lambda r: (r[0], r[1])):
        times = [r[2] for r in grp]
        for i in range(len(times) - 1):
            for j in range(i + 1, len(times)):
                try:
                    fmt = '%Y-%m-%d %H:%M:%S.%f' if '.' in times[i] else '%Y-%m-%d %H:%M:%S'
                    t1 = datetime.datetime.strptime(times[i], fmt)
                    t2 = datetime.datetime.strptime(times[j], fmt)
                    sp = (t2 - t1).total_seconds()
                    if 0.63 <= sp <= 38.9:
                        pairs.append((key[1], key[0], times[i], round(sp, 1)))
                except Exception:
                    pass

    print()
    if pairs:
        print(f'  S-P pairs in TT range (0.6–38.9s): {len(pairs)}')
        for p in pairs[:8]:
            print(f'    {p[0]}.{p[1]:<8s}  t={p[2]}  s-p={p[3]}s')
        if len(pairs) > 8:
            print(f'    ... and {len(pairs)-8} more')
    else:
        print('  S-P pairs in TT range              : NONE (no seismic event in current window)')

except Exception as e:
    print(f'  ERROR reading DB: {e}', file=sys.stderr)
PYEOF
else
    echo -e "  ${YELLOW}DB not found: ${ASSOC_DB}${RESET}"
fi
echo

echo -e "${BOLD}==========================================${RESET}"
