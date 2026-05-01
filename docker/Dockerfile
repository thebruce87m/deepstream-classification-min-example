FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings
# Commit: 8ad0349

# PyDS dependencies

#
# 1.2 Base dependencies
#

# To compile bindings on Ubuntu 24.04, Python 3.12 :
RUN apt update && apt install -y \
python3-gi \
python3-dev \
python3-gst-1.0 \
python-gi-dev \
git \
meson \
python3 \
python3-pip \
python3-venv \
cmake \
g++ \
build-essential \
libglib2.0-dev \
libglib2.0-dev-bin \
libgstreamer1.0-dev \
libtool \
m4 \
autoconf \
automake \
libgirepository-2.0-dev \
libcairo2-dev

# Additionally, install PyPA :
RUN pip3 install build

WORKDIR /opt/nvidia/deepstream/deepstream/sources

RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

#
# 1.3 Initialization of submodules
#

# The repository utilizes gst-python and pybind11 submodules. To initializes them, run the following command:
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
RUN git submodule update --init
RUN python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse

#
# 1.4 Installing Gst-python
#

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/3rdparty/gstreamer/subprojects/gst-python/
RUN meson setup build
RUN cd build && ninja && ninja install


#
# 2.1 Quick build 1 (x86-ubuntu-24.04 | python 3.12 | Deepstream 9.0)
#

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/

RUN export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
RUN python3 -m build

#
# 3. Installing the bindings
#

#
# 3.1 Installing the pip wheel
#

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/dist
RUN pip3 install pyds-1.2.3-cp312-cp312-linux_x86_64.whl


# Install the CUDA Python bindings, which are required by the PyDS bindings. This is not included in the base DeepStream image, but is required for the PyDS bindings to work.
RUN pip3 install cuda-python

WORKDIR /opt/nvidia/deepstream/deepstream-9.0/