nvidia-docker run -it -v /home/yuxiang/GitLab/posecnn-pytorch:/home/yuxiang/GitLab/posecnn-pytorch -v /capri/PoseCNN_Dataset/backgrounds:/home/yuxiang/GitLab/posecnn-pytorch/data/backgrounds -v /capri/PoseCNN_Dataset/models:/home/yuxiang/GitLab/posecnn-pytorch/data/models -v /capri/coco:/home/yuxiang/GitLab/posecnn-pytorch/data/coco -v /capri/ShapeNetCore-nomat:/home/yuxiang/GitLab/posecnn-pytorch/data/shapenet --network host nvcr.io/nvidian/robotics/posecnn-pytorch:latest
