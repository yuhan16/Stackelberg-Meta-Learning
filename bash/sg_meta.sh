#!/bin/bash
scn=0   # or change to other scenarios.
dir="logs/meta"
if [ ! -d $dir ]; then
    mkdir $dir
fi

echo "start meta learning..."
fname="meta_scenario${scn}.txt"
python main_meta $scn > $fname
echo "meta learning complete."
