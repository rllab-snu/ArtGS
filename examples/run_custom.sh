docker cp vis_point.py $2:/workspace/4DGaussians &&
docker cp train.py $2:/workspace/4DGaussians && 
docker cp scene $2:/workspace/4DGaussians && 
docker cp decomposition $2:/workspace/4DGaussians &&
docker cp config $2:/workspace/4DGaussians &&
docker exec --workdir /workspace/4DGaussians -it $2 python train.py -s data/custom/$1 --port 6017 --expname "custom/$1" --configs arguments/dnerf/hellwarrior.py && 
docker exec --workdir /workspace/4DGaussians -it $2 python vis_point.py --model_path output/custom/$1 --configs arguments/dnerf/hellwarrior.py &&
docker cp $2:/workspace/4DGaussians/output/custom/$1 ./
