# README

## Build
```bash
cd ray
docker build -t ray-shunfeng .
docker images | grep ray-shunfeng
docker tag ray-shunfeng:latest dengwxn/ray-shunfeng:latest
docker push dengwxn/ray-shunfeng:latest
```

## Run
```bash
docker run -d --name ray-shunfeng-run \
  --shm-size=128gb \
  --gpus all \
  --cap-add SYS_PTRACE \
  -v $(pwd):/app \
  -it ray-shunfeng
docker exec -it ray-shunfeng-run zsh
```

## Clean
```bash
docker stop ray-shunfeng-run
docker rm ray-shunfeng-run
```

## Install
```bash
cd /app/container
./install_container.sh
```
