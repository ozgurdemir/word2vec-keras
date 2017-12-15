FROM gw000/keras:2.1.1-py3-tf-cpu

# install dependencies from debian packages
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    python3-setuptools

RUN easy_install3 pip

ADD requirements.txt .

# install dependencies from python packages
RUN pip --no-cache-dir install -r requirements.txt
