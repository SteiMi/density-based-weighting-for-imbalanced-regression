VERSION="v1.2.0"

docker build . -t INSERT_YOUR_REGISTRY/denseloss:$VERSION
# Push image to our docker registry
docker push INSERT_YOUR_REGISTRY/denseloss:$VERSION
