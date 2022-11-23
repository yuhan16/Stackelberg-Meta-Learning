#!/bin/bash
scn=0   # or change to other scenarios.
dir="logs/nometa"
if [ ! -d $dir ]; then
    mkdir $dir
fi

echo "start individual learning..."
fname="meta_scenario${scn}.txt"
python main_meta $scn > $fname
echo "individual learning complete."
