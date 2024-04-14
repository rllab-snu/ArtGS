docker cp train_seg.py $2:/workspace/4DGaussians && 
docker cp render_seg.py $2:/workspace/4DGaussians && 
docker cp decomposition $2:/workspace/4DGaussians &&
docker cp config $2:/workspace/4DGaussians &&
docker exec --workdir /workspace/4DGaussians -it $2 python train_seg.py -s data/custom/$1 --port 6017 --expname "custom/$1" --configs arguments/dnerf/hellwarrior.py --decomp_configs config/custom/$1.yaml && 
#docker cp 4dgs:/workspace/4DGaussians/output/custom/$1 ./
docker cp $2:/workspace/4DGaussians/output/custom/$1/finearttrainlabel_render ./$1 &&
docker cp $2:/workspace/4DGaussians/output/custom/$1/finearttestlabel_render ./$1
docker cp $2:/workspace/4DGaussians/output/custom/$1/seg_model.th ./$1 &&
docker exec --workdir /workspace/4DGaussians -it $2 python render_seg.py --model_path "output/custom/$1/" --skip_train --configs arguments/dnerf/hellwarrior.py &&
docker cp $2:/workspace/4DGaussians/output/custom/$1/test ./$1 &&
docker cp $2:/workspace/4DGaussians/output/custom/$1/video ./$1
#docker cp 4dgs:/workspace/4DGaussians/output/dnerf/$1/finearttrain_render ./$1 &&
#docker cp 4dgs:/workspace/4DGaussians/output/dnerf/$1/finearttest_render ./$1
