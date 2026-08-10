source=$1 
output=$2 
export CUDA_VISIBLE_DEVICES=$3
logfile=cePCL.txt
output_path=output/ablation
exec > >(tee -a "$logfile") 2>&1
for NAME in Ballroom  Barn  Church  Family  Francis  Horse  Ignatius  Museum
do
    echo "==== Processing $NAME ===="
    python joint_train.py  --source_path $source/$NAME \
                            --model_path $output_path/$NAME  --eval  --n_views 3 \
                            --iterations 10000 --boundary_threshold 0.05 \
                            --size_threshold_rate 1 --sample_pseudo_interval 10

    python render.py  --source_path $source/$NAME  --model_path $output_path/$NAME --iteration 3000
    python render.py  --source_path $source/$NAME  --model_path $output_path/$NAME --iteration 1000
    python render.py  --source_path $source/$NAME  --model_path $output_path/$NAME --iteration 6000
    python render.py  --source_path $source/$NAME  --model_path $output_path/$NAME --iteration 7000
    python render.py  --source_path $source/$NAME  --model_path $output_path/$NAME --iteration 10000

    python metrics.py  --source_path $source  --model_path $output_path/$NAME --iteration 1000
    python script/eval_lerf_mask.py ablation/$NAME
done