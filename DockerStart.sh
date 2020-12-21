#!/bin/bash
#
# This script will start the docker image: usage DockerBuildStart "imageversion"
#
if [ $1 ]
then
    xhost local:docker
    # use this when webcam is connected: --device /dev/video0:/dev/video0
    sudo docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.0 -v /tmp/.X11-unix/:/tmp/.X11-unix -v /home/eevers/nvme/projects/deepstream_fr/logdir:/opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream_fr/logdir deepstream_fr:$1 
else
    echo "please specify imageversion"
fi
