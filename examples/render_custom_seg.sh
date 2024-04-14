docker cp render_seg.py $2:/workspace/4DGaussians && 
docker cp decomposition $2:/workspace/4DGaussians &&
docker exec --workdir /workspace/4DGaussians -it $2 python render_seg.py --model_path "output/custom/$1/" --skip_train --configs arguments/dnerf/hellwarrior.py &&
docker cp $2:/workspace/4DGaussians/output/custom/$1/test ./$1 &&
docker cp $2:/workspace/4DGaussians/output/custom/$1/video ./$1
