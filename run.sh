# python data_process.py --in_dir /home/admin/Pictures/ECG_data/neg/part1 --out_dir ../data/processed_neg/part1 

# python data_process.py --in_dir /home/admin/Pictures/ECG_data/neg/part2 --out_dir ../data/processed_neg/part2 

# python data_process.py --in_dir /home/admin/Pictures/ECG_data/neg/part3 --out_dir ../data/processed_neg/part3 

# python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.001 --task "class" --model 'resnet18' --training True --bs 64  
# # --pretrained True --continue_train True

# ======== pretrained 
# resnet18 
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0001 --task "class" --model 'resnet18' --training True --bs 64  --pretrained True --n_run 1
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0001 --task "class" --model 'resnet18' --training True --bs 64  --pretrained True --n_run 2
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0001 --task "class" --model 'resnet18' --training True --bs 64  --pretrained True --n_run 3

# ======== from scratch 
# resnet18 
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0005 --task "class" --model 'resnet18' --training True --bs 64  --n_run 1
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0005 --task "class" --model 'resnet18' --training True --bs 64  --n_run 2
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0005 --task "class" --model 'resnet18' --training True --bs 64  --n_run 3

# ======== pretrained 
# resnet50
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0001 --task "class" --model 'resnet50' --training True --bs 32  --pretrained True --n_run 1
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0001 --task "class" --model 'resnet50' --training True --bs 32  --pretrained True --n_run 2
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0001 --task "class" --model 'resnet50' --training True --bs 32  --pretrained True --n_run 3

# ======== from scratch 
# resnet50
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0005 --task "class" --model 'resnet50' --training True --bs 32  --n_run 1
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0005 --task "class" --model 'resnet50' --training True --bs 32  --n_run 2
python main.py --main_path "/home/chain/gpu" --n_epoch 50 --lr 0.0005 --task "class" --model 'resnet50' --training True --bs 32  --n_run 3


