python setup.py bdist_wheel

cp dist/torch_radon-1.0.0-cp38-cp38-linux_x86_64.whl docker/

cd docker
docker build . -t matteoronchetti/torch-radon

cd ..