time=$(date '+%Y%m%d%H%M%S')

## input file
dataset_dir="/home/ykhsieh/CV/final/dataset/"
label_data="/home/ykhsieh/CV/final/dataset/conf.json"

## output file
learning_curv_dir="../log-${time}/curv"
check_point_root="../log-${time}/checkpoints"
log_root="../log-${time}"
batch_size=16
lr=0.001
num_epochs=200
m1=15
m2=40
m3=60
train_val=0

if [ ! -d "../train-${time}" ]; then
mkdir -p ../log-${time}/{checkpoints,python_backups,curv}
fi

cp trainer.py ../log-${time}/python_backups
cp module.py ../log-${time}/python_backups
cp data.py ../log-${time}/python_backups
cp run_train.sh ../log-${time}/python_backups 

train_bin="python3 train.py "
CUDA_VISIBLE_DEVICES=1 $train_bin \
--dataset_dir ${dataset_dir} \
--label_data ${label_data} \
--learning_curv_dir ${learning_curv_dir} \
--check_point_root ${check_point_root} \
--log_root ${log_root} \
--batch_size ${batch_size} \
--lr ${lr} \
--num_epochs ${num_epochs} \
--train_val ${train_val} \
--milestones ${m1} ${m2} ${m3} \
