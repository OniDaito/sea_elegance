#!/bin/bash
if [ $# < 2 ]
  then
    echo "Usage eval.sh run_dir dataset_dir"
    exit 1
fi

python eval.py --load $1 --data $2 --save
mv eval_* $1/.

# Now select an image from the validation dataset and do the compares
# In fact, do the entire set
count=0
tail -n+2 $1/dataset_valid.csv | grep --line-buffered '.*' | while read LINE0
do
    IFS=", " read -a fnames <<< $LINE0
    python viz/viz.py --image $2/${fnames[0]} --cutoff 400 --savepath $1/eval_input_$count.ply --rez 400
    python viz/viz.py --image $2/${fnames[1]} --cutoff 1 --savepath $1/eval_mask_$count.ply --rez 400
    python viz/viz.py --image $1/eval_$count.fits --cutoff 1 --savepath $1/eval_pred_$count.ply --rez 400
    count=$((count+1))
done

# One can now use vedo to visualise the 3 ply files
# vedo eval_input_5.ply eval_mask_5.ply eval_pred_5.ply -a 0.3