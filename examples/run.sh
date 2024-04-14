docker cp train_seg.py $2:/workspace/4DGaussians && 
docker cp decomposition $2:/workspace/4DGaussians &&
docker cp config $2:/workspace/4DGaussians &&
docker exec --workdir /workspace/4DGaussians -it $2 python train_seg.py -s data/dnerf/$1 --port 6017 --expname "dnerf/$1" --configs arguments/dnerf/$1.py && 
#docker cp $2:/workspace/4DGaussians/output/dnerf/$1 ./
docker cp $2:/workspace/4DGaussians/output/dnerf/$1/finearttrainlabel_render ./$1 &&
docker cp $2:/workspace/4DGaussians/output/dnerf/$1/finearttestlabel_render ./$1  &&
docker cp $2:/workspace/4DGaussians/output/dnerf/$1/seg_model.th ./$1
#docker cp 4dgs:/workspace/4DGaussians/output/dnerf/$1/finearttrain_render ./$1 &&
#docker cp 4dgs:/workspace/4DGaussians/output/dnerf/$1/finearttest_render ./$1
