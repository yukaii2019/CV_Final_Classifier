time=$(date '+%Y%m%d%H%M%S')

## input file
dataset_path="/home/ykhsieh/CV/final/dataset/"
classifier_checkpoint="/home/ykhsieh/CV/final/classifier/log-20230603180236/checkpoints/model_best_9998.pth"

## output file
output_path="/home/ykhsieh/CV/final/output9"

bin="python3 inference.py "
CUDA_VISIBLE_DEVICES=1 $bin \
--dataset_path ${dataset_path} \
--classifier_checkpoint ${classifier_checkpoint} \
--output_path ${output_path} \