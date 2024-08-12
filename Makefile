# Get version of CUDA and enable it for compilation if CUDA > 11.0
# This solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
# and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84
# when running in Docker
# Check if nvcc is installed
NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
	# NVCC not found
	USE_CUDA := 0
	NVCC_VERSION := "not installed"
else
	NVCC_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9.]+')
	USE_CUDA := $(shell echo "$(NVCC_VERSION) > 11" | bc -l)
endif

# Add the list of supported ARCHs
ifeq ($(USE_CUDA), 1)
	TORCH_CUDA_ARCH_LIST := "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
	BUILD_MESSAGE := "I will try to build the image with CUDA support"
else
	TORCH_CUDA_ARCH_LIST :=
	BUILD_MESSAGE := "CUDA $(NVCC_VERSION) is not supported"
endif


build-image:
	@echo $(BUILD_MESSAGE)
	docker build --build-arg USE_CUDA=$(USE_CUDA) \
	--build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
	-t artgs:v0 .
	docker run -d --gpus all -it --rm --net=host \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "${PWD}":/home/appuser/ArtGS \
	-w /home/appuser/ArtGS \
	-e DISPLAY=$DISPLAY \
	--name=artgs \
	--ipc=host artgs:v0
	docker exec -it socc sh -c "pip install -e submodules/depth-diff-gaussian-rasterization"
	docker exec -it socc sh -c "pip install -e submodules/simple-knn"
	docker commit artgs artgs:latest

run:
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$(DISPLAY) -e USER=$(USER) \
	-e runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all \
	-v "${PWD}":/home/appuser/ArtGS \
	-w /home/appuser/ArtGS \
	--shm-size 8G \
	--net host --gpus all --privileged --name artgs artgs:latest /bin/bash