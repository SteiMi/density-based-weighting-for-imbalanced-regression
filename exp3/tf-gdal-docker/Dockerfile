FROM tensorflow/tensorflow:1.13.1-gpu

RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable

# Existing binutils causes a dependency conflict, correct version will be installed when GDAL gets intalled
RUN apt-get remove -y binutils

# Install GDAL dependencies
RUN apt-get update && \
    apt-get install -y \
    libgdal-dev \
    g++ \
    python-tk \
    libsm6 \
    libxext6 \
    libxrender-dev

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install correct GDAL version depending on the one installed in the container
RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

CMD [ "echo", "You should add your code and change this command!" ]
