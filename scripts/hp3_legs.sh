#!/bin/bash
# build all legs objects; logs in out/hp3_legs_<name>.log ; usage: scripts/hp3_legs.sh [names...]
PY=${PY:-/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python}
cd "$(dirname "$0")/.."
ALL="trousers"; for s in l r; do for n in shaft tab band1 band2 toe foot sole; do ALL="$ALL boot_${n}_$s"; done; done
if [ -z "$NOGROW" ]; then $PY -m retopo.seeds grow --out survey/labels_legs.npz >/dev/null 2>&1; fi
for o in ${@:-$ALL}; do
  case $o in trousers) E=3.5;; *) E=2.5;; esac
  $PY -m retopo.fromseg $o --edge $E --labels survey/labels_legs.npz > out/hp3_legs_$o.log 2>&1 &
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 5; done
done
wait
$PY -m scripts.review3 out/hp3_legs.png 0.465 0.1 0.15 0.25 front,fl,left,back hipoly3 $ALL
