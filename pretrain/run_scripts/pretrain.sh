CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29576 --nnodes 1 --nproc_per_node 4 main.py --enable_fsdp \
	--output_dir /mnt/data/ssl/output/pretrain \
	--batch_size_training 72