#python train.py -s data/dnerf/$1 --expname "dnerf/$1" --configs arguments/dnerf/$1.py 
python train_seg.py -s data/dnerf/$1 --port 6017 --expname "dnerf/$1" --configs arguments/dnerf/$1.py --decomp_configs config/dnerf/$1.yaml && 
python render_seg.py --model_path "output/dnerf/$1/" --skip_train --configs arguments/dnerf/$1.py

#docker cp ../ArtGS $2:/workspace &&
#docker exec --workdir /workspace/ArtGS -it $2 python train_seg.py -s data/dnerf/$1 --port 6017 --expname "dnerf/$1" --configs arguments/dnerf/$1.py --decomp_configs config/dnerf/$1.yaml && 
#docker exec --workdir /workspace/ArtGS -it $2 python render_seg.py --model_path "output/dnerf/$1/" --skip_train --configs arguments/dnerf/$1.py &&
#docker cp $2:/workspace/ArtGS/output/dnerf/$1/finearttestlabel_render ./$1 &&
#docker cp $2:/workspace/ArtGS/output/dnerf/$1/finearttrainlabel_render ./$1 &&
#docker cp $2:/workspace/ArtGS/output/dnerf/$1/test ./$1 &&
#docker cp $2:/workspace/ArtGS/output/dnerf/$1/video ./$1