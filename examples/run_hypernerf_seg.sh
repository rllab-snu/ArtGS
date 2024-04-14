docker cp train_seg.py $3:/workspace/4DGaussians && 
docker cp render_seg.py $3:/workspace/4DGaussians && 
docker cp decomposition $3:/workspace/4DGaussians &&
docker cp config $3:/workspace/4DGaussians &&
docker exec --workdir /workspace/4DGaussians -it $3 python train_seg.py -s data/hypernerf/$1/$2 --port 6017 --expname "hypernerf/$2" --configs arguments/hypernerf/$2.py && 
#docker cp 4dgs:/workspace/4DGaussians/output/hypernerf/$1 ./
docker cp $3:/workspace/4DGaussians/output/hypernerf/$2/finearttrainlabel_render ./$2 &&
docker cp $3:/workspace/4DGaussians/output/hypernerf/$2/finearttestlabel_render ./$2
docker cp $3:/workspace/4DGaussians/output/hypernerf/$2/seg_model.th ./$2 &&
docker exec --workdir /workspace/4DGaussians -it $3 python render_seg.py --model_path "output/hypernerf/$2/" --skip_train --configs arguments/hypernerf/$2.py &&
docker cp $3:/workspace/4DGaussians/output/hypernerf/$2/test ./$2 &&
docker cp $3:/workspace/4DGaussians/output/hypernerf/$2/video ./$2
#docker cp 4dgs:/workspace/4DGaussians/output/dnerf/$1/finearttrain_render ./$1 &&
#docker cp 4dgs:/workspace/4DGaussians/output/dnerf/$1/finearttest_render ./$1
