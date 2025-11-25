CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
--eval \
--ckpt_path ./logs/ScanSSC_SemanticKITTI/checkpoints/best.ckpt \
--config_path ./configs/semantickitti_ScanSSC.py \
--log_folder semantickitti_ScanSSC_eval \
--seed 7240 \
--log_every_n_steps 50 \
--save_path pred_semantickitti