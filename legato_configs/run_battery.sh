#!/bin/bash
cd ~/proj/vllm-omni-legato
mkdir -p battery
declare -a TEXTS
TEXTS[0]="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
TEXTS[1]="Under the low light of the winter morning the harbor slowly filled with small boats, each one leaving a thin white line across the grey water, and the fishermen called to one another over the sound of the engines."
TEXTS[2]="When the orchestra finally settled and the conductor raised the baton, the hall fell into a deep silence that seemed to stretch for minutes, and then the first low notes of the cellos moved through the room."
TEXTS[3]="Seventy three percent of the survey respondents reported that their commute changed substantially after the new line opened in twenty twenty four, with average travel times falling from forty one minutes to twenty six."
TEXTS[4]="She spoke quickly and quietly, almost in a whisper, listing the names of the stations one after another, and the child repeated each one, stumbling over the longer words but never losing the thread of the game."
TEXTS[5]="The recipe calls for two cups of flour, a pinch of salt, three eggs beaten until pale, and butter melted slowly over low heat, folded together in wide circles until the batter turns smooth and heavy."
for i in 0 1 2 3 4 5; do
  T="${TEXTS[$i]} ${TEXTS[$i]} ${TEXTS[$i]} ${TEXTS[$i]}"
  .venv/bin/python legato_bench_async.py --concurrency 48 --text "$T" --out battery/c48_t$i.ndjson --metrics-out battery/c48_t$i.metrics.ndjson > battery/c48_t$i.log 2>&1
  sleep 10
done
echo DONE > battery.marker
