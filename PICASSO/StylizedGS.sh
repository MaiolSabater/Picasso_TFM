DATA_TYPE=$1
SCENE=$2
STYLE=$3

ckpt_gs=output/ckpt_gs/${DATA_TYPE}/${SCENE}
ckpt_stylegs=output/ckpt_stylegs/${DATA_TYPE}/${SCENE}_${STYLE}_trial_2222_14
data_dir=/data/storage/users/msabater/datasets/${DATA_TYPE}/${SCENE}
style_img=/data/storage/users/msabater/datasets/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_gs}/point_cloud/iteration_30000/point_cloud.ply" ]]; then
    python train.py -s ${data_dir} \
                  -m ${ckpt_gs}
fi

python train_style.py -s ${data_dir} \
                -m ${ckpt_stylegs} \
                --point_cloud ${ckpt_gs}/point_cloud/iteration_30000/point_cloud.ply \
                --style ${style_img} \
                --gt_l_strength 0 \
                --gt_theme_size 5 \
                --gt_theme_source gaussians \
                --direct_match \
                --margin_pull \
                --margin_base_scale 0.1\
                --gt_match_by unique \
                --recolor_debug
                
                

                # --recolor_space rgb_azimuth \
                # --n_colors 8 \
                # --style_n_colors 8 \
                # --xy_weight 1.0 \      
                #--gt_recolor_space ab \
                #--gt_power 2
                #--gt_theme_size 5
                # --mask_dir ${data_dir}/masks \
                # --second_style datasets/styles/12.jpg \
                # --scale_level 0 \
                

python render.py -m ${ckpt_stylegs} \
                        --point_cloud ${ckpt_stylegs}/point_cloud/iteration_31500/point_cloud.ply \
                        --video --eval --fps 30
