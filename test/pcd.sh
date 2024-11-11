# xvfb-run -a python get_pcd.py \
#     --data_dir ../data_collection/data/test_data \
#     --record_name 152_Faucet_3_pulling_42 \
#     --use_mask True

OUPUT_DIR='./pcds'
python test_entireprocess_in_sapien.py \
  --data_dir ../data_collection/data/test_data \
  --num_processes 20 \
  --out_dir "$OUPUT_DIR" \
  --use_mask True 