export OMP_NUM_THREADS=24
python3 eval.py   --num_warmup 100 --num_steps 5000 --pb_model_file model/resnet_v1_101.npu.pb --validation_path imagenet_val/tf_record/ 
