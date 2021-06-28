cd ..
cp requirements.txt evaluate_k8s/requirements.txt
docker build -f evaluate_k8s/Dockerfile . -t deepsd-evaluate
# Push image to our docker registry
docker tag deepsd-evaluate INSERT_YOUR_REGISTRY/deepsd-evaluate
docker push INSERT_YOUR_REGISTRY/deepsd-evaluate
cd evaluate_k8s
