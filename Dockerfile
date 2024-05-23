## Set Base Stage
FROM rocm/dev-ubuntu-20.04:6.0.2 AS rbm-base

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
apt-utils \
ssh \
software-properties-common \
apt-transport-https ca-certificates gnupg software-properties-common wget \
bash-completion \
gcc-9 g++-9 gcc-9-base \
libopenmpi-dev \
&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
&& update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt-get update && \
apt-get install -y python3.10 && \
apt-get install -y python3.10 && \
apt-get install -y python3.10-distutils && \
apt-get install -y python3.10-dev && \
apt-get install -y libjpeg-dev && \
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 100 && \
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 100 && \
wget https://bootstrap.pypa.io/get-pip.py && \
python3 get-pip.py && \
rm -rf get-pip.py


RUN apt-get update && \
apt-get install -y cmake && \
apt-get install -y git

RUN pip3 install numpy --no-cache-dir && \
pip3 install scipy --no-cache-dir && \
pip3 install matplotlib --no-cache-dir && \
pip3 install tqdm --no-cache-dir && \
pip3 install wheel --no-cache-dir && \
pip3 install setuptools --no-cache-dir

RUN pip3 install  torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0 --no-cache-dir

WORKDIR /content
RUN git clone https://github.com/fvicini/CppToPython.git

WORKDIR /content/CppToPython
RUN git submodule init
RUN git submodule update

WORKDIR /content/CppToPython/externals
RUN cmake -DINSTALL_VTK=OFF -DINSTALL_LAPACK=ON ../gedim/3rd_party_libraries && make -j4

WORKDIR /content/CppToPython/release
RUN cmake -DCMAKE_PREFIX_PATH="/content/CppToPython/externals/Main_Install/eigen3;/content/CppToPython/externals/Main_Install/triangle;/content/CppToPython/externals/Main_Install/tetgen;/content/CppToPython/externals/Main_Install/googletest;/content/CppToPython/externals/Main_Install/lapack" ../ && make -j4 GeDiM4Py

WORKDIR /root
ENV PYTHONPATH="/content/CppToPython"
