mkdir -p "./checkpoints"

MODEL1_CKPT="./checkpoints/ss_model=resunet30,querynet=at_soft,data=full,devices=8,step=1000000.ckpt"
MODEL1_YAML="./checkpoints/ss_model=resunet30,querynet=at_soft,data=full.yaml"
MODEL2_CKPT="./checkpoints/ss_model=resunet30,querynet=emb,data=balanced,devices=1,steps=1000000.ckpt"

URL_MODEL1_CKPT="https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull%2Cdevices%3D8%2Cstep%3D1000000.ckpt"
URL_MODEL1_YAML="https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull.yaml?download=true"
URL_MODEL2_CKPT="https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Demb%2Cdata%3Dbalanced%2Cdevices%3D1%2Csteps%3D1000000.ckpt"

if [ ! -f "$MODEL1_CKPT" ]; then
    wget -O "$MODEL1_CKPT" "$URL_MODEL1_CKPT"
fi

if [ ! -f "$MODEL1_YAML" ]; then
    wget -O "$MODEL1_YAML" "$URL_MODEL1_YAML"
fi

if [ ! -f "$MODEL2_CKPT" ]; then
    wget -O "$MODEL2_CKPT" "$URL_MODEL2_CKPT"
fi

PYTHONPATH=~/source/uss CUDA_VISIBLE_DEVICES=0 python uss/inference.py \
    --audio_path=./resources/harry_potter.flac \
    --levels 1 2 3 \
    --config_yaml="$MODEL1_YAML" \
    --checkpoint_path="$MODEL1_CKPT"
