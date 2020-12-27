FROM nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-samples
# this is the deepstream version with the samples.
# WORKDIR /opt/nvidia/deepstream/deepstream-5.0/sources
# upgrade stuff and install required programs
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install python3-pip && \
    # install Gst-python
    apt-get -y install python3-gi python3-dev python3-gst-1.0 && \
    # install additional items
    apt-get install -y --no-install-recommends \
    wget \
    cmake \
    unzip \
    curl \
    iputils-ping \
    nano
# download and compile dlib and OpenCV with CUDA
ARG OPENCV_VERSION=4.5.0
RUN cd /opt/nvidia && \
    # show cuda version
    nvcc --version && \
    apt-get install -y \
    # python3-pip \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
    # and the following just to be sure for opencv2 install
    ffmpeg \
    libtbb2 \
    gfortran \
    apt-utils \
    pkg-config \
    checkinstall \
    qt5-default \
    build-essential \
    libopenblas-base \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    # libavcodec58 this one seems to not exist
    libavcodec-dev \
    # libavformat58 this one seems to not exist
    libavformat-dev \
    libavutil-dev \
    # libswscale5 this one seems not to exist
    libswscale-dev \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libdc1394-22 \
    libdc1394-22-dev \
    libxine2-dev \
    libv4l-dev \
    libgstreamer1.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libglew-dev \
    libpostproc-dev \
    libeigen3-dev \
    libtbb-dev \
    zlib1g-dev \
    libsm6 \
    libxext6 \
    libxrender1
# upgrade cuda
RUN cd /opt/nvidia && \
    mkdir cuda1120 && cd cuda1120 && \
    # remove old version first => not
    # apt-get --purge remove cuda-10.2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/cuda-ubuntu1804.pin && \
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_arm64.deb && \
    apt-get -y install gnupg2 && \
    dpkg -i cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_arm64.deb && \
    apt-key add /var/cuda-repo-ubuntu1804-11-2-local/7fa2af80.pub && \
    apt-get update  && \
    apt-get -y install cuda && \
    export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}$ && \
    export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} && \
    nvcc --version
# install dlib
RUN cd /opt/nvidia && \
    mkdir dlib1921 && \
    cd dlib1921 && \
    export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}$ && \
    export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} && \
    wget -O dlib.zip https://github.com/davisking/dlib/archive/master.zip -O dlib.zip --progress=bar:force:noscroll && \
    unzip dlib.zip && \
    cd dlib-master && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . && \
    cd ../ && \
    python3 setup.py install
# here we should install opencv, but first get dlib comiled with CUDA support
# install bindings for DeepStream Metadata libraries
RUN cd /opt/nvidia/deepstream/deepstream/lib && \
    export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}$ && \
    export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} && \
    python3 setup.py install && \
    # and install other programs
    apt-get install -y \
    python3-numpy \
    libgstrtspserver-1.0-0 gstreamer1.0-rtsp \
    libgirepository1.0-dev \
    gobject-introspection gir1.2-gst-rtsp-server-1.0 \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libgstrtspserver-1.0-dev libx11-dev
# install demo python apps
RUN cd /opt/nvidia/deepstream/deepstream-5.0 && \
    # sources dir is already part of the samples image but not in the base image: mkdir sources && \
    cd sources && \
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps && \
    cd .. && \
    # make the gstreamer for python
    cd /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream-test3 && \
    apt-get -y install autoconf libtool libgstreamer1.0-dev python-gi-dev && \
    export GST_LIBS="-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0" && \
    export GST_CFLAGS="-pthread -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include" && \
    git clone https://github.com/GStreamer/gst-python.git && \
    cd gst-python && \
    git checkout 1a8f48a && \
    ./autogen.sh PYTHON=python3 && \
    ./configure PYTHON=python3 && \
    make && \
    make install && \
    cd ..
# and our own additions to allow python face recognition
RUN apt-get install -y \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran
#    pip3 install face_recognition
# cleanup
RUN rm -rf /opt/nvidia/dlib1921 && \
    rm -rf /opt/nvidia/cuda1120
# get our software ready and at the correct place
WORKDIR /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream_fr
COPY    ./src/ .
RUN cd /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream_fr && \
    mkdir logdir
# Call default command and show what we have are running
RUN python3 --version