python experiments/margin_pull_sweep_multi.py \
  --scenes truck:output/ckpt_gs/tandt/truck/point_cloud/iteration_30000/point_cloud.ply \
           family:output/ckpt_gs/tandt/family/point_cloud/iteration_30000/point_cloud.ply \
           m60:output/ckpt_gs/tandt/m60/point_cloud/iteration_30000/point_cloud.ply \
           flower:output/ckpt_gs/llff/flower/point_cloud/iteration_30000/point_cloud.ply\
  --styles style14:/data/storage/users/msabater/datasets/styles/9.jpg \
           style15:/data/storage/users/msabater/datasets/styles/14.jpg \
           style16:/data/storage/users/msabater/datasets/styles/19.jpg \
           style17:/data/storage/users/msabater/datasets/styles/17.jpg \
           style32:/data/storage/users/msabater/datasets/styles/32.jpg \
           style34:/data/storage/users/msabater/datasets/styles/34.jpg \
           style122:/data/storage/users/msabater/datasets/styles/122.jpg \
  --output_dir output/experiments/margin_sweep \
  --match_by hue
