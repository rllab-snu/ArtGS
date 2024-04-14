docker exec --workdir /workspace/4DGaussians -it $3 python train.py -s  data/hypernerf/$1/$2 --port 6017 --expname "hypernerf/$2" --configs arguments/hypernerf/$2.py  && 
docker exec --workdir /workspace/4DGaussians -it $3 python vis_point.py --model_path output/hypernerf/$2 --configs arguments/hypernerf/$2.py &&
docker cp $3:/workspace/4DGaussians/output/hypernerf/$2 ./
