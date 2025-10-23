#export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH

train_data=igrot
eval_data=fiq # fiq, cirr, circo, dtin, sketchy, tuberlin, quickdraw
model=blip # clip_base, clip_large, blip(base), blip_flickr
encoder=both
batch_size=16
epochs=2
target_type=fuse
llava=with
note=${llava}_llava_${target_type}
comment=_5k_${note}_${encoder}_fuse_extended
name=${model}_${epochs}_epo$comment

save_path=./ckpt/final_model_$name
eval_load_path=./ckpt/final_model_$name.pth
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=25033 \
	main.py \
	--training True \
	--dataset $train_data \
	--encoder $encoder \
	--model $model \
	--llava $llava \
	--type ${target_type} \
	--batch-size $batch_size \
	--learning-rate 1e-4 \
	--num-epochs $epochs \
    --save-path ${save_path} \
	--comment $name \
	--inference True \
	--val_dataset $eval_data \
	--val_load_path ${eval_load_path} \
	--submission_name ${name}
