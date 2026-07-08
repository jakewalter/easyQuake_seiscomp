#!/bin/bash
# archive_cleanup.sh — delete SeisComP SDS archive files older than RETAIN_DAYS.
#
# Why not -mtime: gap_recovery backfills old day-files, refreshing their mtime.
# Instead we parse the YEAR.JULDAY encoded in every SDS filename.
#
# SDS layout:  ARCHIVE/YEAR/NET/STA/CHAN.D/NET.STA.LOC.CHAN.D.YEAR.JULDAY
#              depth:   1    2   3   4      5 (actual data files)

ARCHIVE=/home/jwalter/seiscomp/var/lib/archive
RETAIN_DAYS=${1:-30}

# Cutoff as zero-padded YYYYDDD string, e.g. "2026094"
CUTOFF=$(date -d "${RETAIN_DAYS} days ago" +%Y%j)

echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] archive_cleanup: retain=${RETAIN_DAYS}d cutoff=${CUTOFF}"

deleted=0
while IFS= read -r f; do
    base="${f##*/}"             # NET.STA.LOC.CHAN.D.YEAR.JULDAY
    jday="${base##*.}"          # JULDAY (3-digit, e.g. 094)
    tmp="${base%.*}"
    year="${tmp##*.}"           # YEAR (4-digit, e.g. 2025)
    # Lexicographic compare works because both are zero-padded to 7 chars
    if [[ "${year}${jday}" < "${CUTOFF}" ]]; then
        rm -f "$f"
        (( deleted++ ))
    fi
done < <(find "$ARCHIVE" -mindepth 5 -maxdepth 5 -type f)

echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] archive_cleanup: deleted ${deleted} file(s)"

# Remove empty channel/station/net/year directories (deepest first)
find "$ARCHIVE" -mindepth 1 -maxdepth 4 -type d -empty -delete

echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] archive_cleanup: done"
