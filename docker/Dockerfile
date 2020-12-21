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
# install bindings for DeepStream Metadata libraries
RUN cd /opt/nvidia/deepstream/deepstream/lib && \
    python3 setup.py install && \
    # and install other programs
    apt-get install -y \
    python3-opencv \
    python3-numpy \
    libgstrtspserver-1.0-0 gstreamer1.0-rtsp \
    libgirepository1.0-dev \
    gobject-introspection gir1.2-gst-rtsp-server-1.0 \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libgstrtspserver-1.0-dev libx11-dev
# install demo python apps
RUN cd /opt/nvidia/deepstream/deepstream-5.0 && \
    # sources dir is already part of the samples image but not the base image: mkdir sources && \
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
    gfortran && \
    pip3 install face_recognition
# get our software ready and at the correct place
WORKDIR /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream_fr
COPY    ./src/ .
RUN cd /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream_fr && \
    mkdir logdir
# Call default command and show what we have are running
RUN python3 --version && \
    python3 -c "import cv2 ; print(cv2.__version__)" && \
    python3 -c "import cv2 ; print(cv2.getBuildInformation())"
