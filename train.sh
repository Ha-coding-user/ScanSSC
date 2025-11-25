CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
--config_path configs/semantickitti_ScanSSC.py \
--log_folder semantickitti_ScanSSC \
--seed 7240 \
--log_every_n_steps 50 
