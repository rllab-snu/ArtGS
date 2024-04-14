docker cp vis_point_seg.py $2:/workspace/4DGaussians && 
docker cp decomposition $2:/workspace/4DGaussians &&
docker exec --workdir /workspace/4DGaussians -it $2 python vis_point_seg.py --model_path output/dnerf/$1 --configs arguments/dnerf/$1.py &&
docker cp $2:/workspace/4DGaussians/output/dnerf/$1/point_pertimestamp ./$1 &&
docker cp $2:/workspace/4DGaussians/output/dnerf/$1/point_pertimestamp_numpy ./$1
