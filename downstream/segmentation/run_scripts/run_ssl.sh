#task_name="Task03_Liver"
task_name="Task06_Lung"
#task_name="Task07_Pancreas"
#task_name="Task08_HepaticVessel"
#task_name="Task09_Spleen"
#task_name="Task10_Colon"
#task_name="LA_Seg"

gpus="0"
cache_num=500
batch_size=4

MSD_data_base='/mnt/data/ssl/data/decathlon'
LA_Seg_data_base='/mnt/data/ssl/data/LA_Seg/data'
save_base='/mnt/data/ssl/output/segmentation'
pretrain_path='/mnt/data/ssl/ssl_checkpoint.pth'

CUDA_VISIBLE_DEVICES=$gpus python main.py --distributed \
	--task_name $task_name --batch_size $batch_size --cache_num $cache_num \
	--MSD_data_base $MSD_data_base --LA_Seg_data_base $LA_Seg_data_base --save_base $save_base \
	--pretrain_path $pretrain_path