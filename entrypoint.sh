apt-get update
apt-get install -y python3-pip

/usr/bin/python3 -m pip install awscli_plugin_endpoint

# install tensorly for compression experiments
pip install git+https://github.com/tensorly/tensorly git+https://github.com/tensorly/torch
pip install s3fs

export PYTHONPATH=$(pwd)/../earth2/external/era5_wind:$PYTHONPATH
export PYTHONPATH=$(pwd)/../earth2/explore/nbrenowitz:$PYTHONPATH
