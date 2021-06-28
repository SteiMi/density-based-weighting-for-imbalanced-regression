docker build tf-gdal-docker -t INSERT_YOUR_REGISTRY/tf-gdal
docker push INSERT_YOUR_REGISTRY/tf-gdal

cp requirements.txt requirements-docker/requirements.txt
docker build requirements-docker -t INSERT_YOUR_REGISTRY/deepsd-requirements
docker push INSERT_YOUR_REGISTRY/deepsd-requirements

docker build . -t deepsd
# Push image to our docker registry
docker tag deepsd INSERT_YOUR_REGISTRY/deepsd
docker push INSERT_YOUR_REGISTRY/deepsd
