gpu_id=0
task_id='RICORD'
reload_from_pretrained=True


data_path='/mnt/data/ssl/data/RICORD/save/RICORD_nii_resize'
snapshot='/mnt/data/ssl/output/RICORD/'
pretrained_path='/mnt/data/ssl/ssl_checkpoint.pth'
snapshot_dir=$snapshot$task_id


CUDA_VISIBLE_DEVICES=$gpu_id python -u main.py \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--pretrained_path=$pretrained_path \
--reload_from_pretrained=$reload_from_pretrained
