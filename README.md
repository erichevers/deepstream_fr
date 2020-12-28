# Deepstream Face Recognition
This Face Recognition program for the Xavier NX is based on NVIDIA DeepStream

This is work in progress, but the basics are working

# Setup:
Get NVIDIA Jetson Xavier NX out of the box
setup according to: https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#prepare
upgrade:
- sudo apt-get update
- sudo apt-get upgrade
- Select x server default display manager: lightDM
- sudo apt autoremove

added nvme disk according to: https://desertbot.io/blog/jetson-xavier-nx-ssd-setup

install jtop:
- sudo apt-get install python3-pip
- sudo -H pip install -U jetson-stats

# Setup project environment and run the docker to run the application
create working directory on host:
- cd ~/nvme
- sudo mkdir projects
- sudo chown 'your name' projects
- cd projects
- setup git stuff (document later): git clone ....
- cd ~/nvme/projects/deepstream_fr
- mkdir logdir
- cd logdir
- mkdir known_faces
(copy your own known faces .pkl file to this directory)
- mkdir unknown_faces
- mkdir test_images
(copy some test images to this directory if you like)
- mkdir logs
- cd ..
- ./DockerBuilsStart.sh 'your image version/tag'

After a while you should be on the prompt of the docker image. You can run the application: 
- python3 deepstream_fr.py 'your rstp video stream or your file' 'or multiple streams' 'directory to store frames'

# some more info
The docker image will pull xxx document later xxx

The ./DockerBuildStart.sh will create the docker image and start this.

The ./DockerStart.sh will just start the docker image

The docker image when started will be in the xxx directory

it will also have a ../logdir direcory which is mounted on the Xavier NX host and can be found at /opt/deepstream_fr/logdir and can be used to store the information that is private or which you want to access via a standard terminal or via ftp, for example to look at the logfiles stored in logdir/logs or the images stored in logdir/unknown_faces.

# Train the face recognition
The application requires a file to check if it is a known face. To store known faces the learn_fr.py program is included. This program takes photo's/images from a directory, encode these for face recognition and insert these in a .pkl file. Each photo may contain only one person and the name of the photo needs to be the name of the person, in the format: firstname lastname, optionally followed by -n where n is the sequence number of the photo to allow multiple photo's per person

# License
The application is released under version 2.0 of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0).
