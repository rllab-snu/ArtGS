docker cp render_seg.py $2:/workspace/4DGaussians && 
docker cp decomposition $2:/workspace/4DGaussians &&
docker exec --workdir /workspace/4DGaussians -it $2 python render_seg.py --model_path "output/dnerf/$1/" --skip_train --configs arguments/dnerf/$1.py &&
docker cp $2:/workspace/4DGaussians/output/dnerf/$1/test ./$1 &&
docker cp $2:/workspace/4DGaussians/output/dnerf/$1/video ./$1
