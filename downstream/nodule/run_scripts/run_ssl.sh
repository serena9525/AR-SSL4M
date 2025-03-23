gpu_id=0
root='/mnt/data/ssl/data/nodule_128'
output_root='/mnt/data/ssl/output/nodule'
pretrain_path='/mnt/data/ssl/ssl_checkpoint.pth'


CUDA_VISIBLE_DEVICES=$gpu_id python train_and_eval.py \
	--root $root \
	--output_root $output_root \
	--pretrain_path $pretrain_path

