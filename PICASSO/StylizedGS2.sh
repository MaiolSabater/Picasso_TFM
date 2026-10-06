DATA_TYPE=$1
SCENE=$2
STYLE=$3

ckpt_gs=output/ckpt_gs/${DATA_TYPE}/${SCENE}
ckpt_stylegs=output/ckpt_stylegs/${DATA_TYPE}/${SCENE}_${STYLE}_original
data_dir=/data/storage/users/msabater/datasets/${DATA_TYPE}/${SCENE}
style_img=/data/storage/users/msabater/datasets/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_gs}/point_cloud/iteration_30000/point_cloud.ply" ]]; then
    python train.py -s ${data_dir} \
                  -m ${ckpt_gs} \
                  --disable_viewer
fi

                

# python render.py -m ${ckpt_stylegs} \
#                         --point_cloud ${ckpt_stylegs}/point_cloud/iteration_31500/final_point.ply \
#                         --video --eval --fps 30
