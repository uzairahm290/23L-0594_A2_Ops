#!/usr/bin/env bash
# Task 6.1 - Docker warm-up with a ready-made image (nginx). Screenshot after each STEP.
set -x
mkdir -p ~/vg-warmup && echo "<h1>23L-0594 docker warm-up</h1>" > ~/vg-warmup/index.html

: STEP 1 pull and run, mapping host 8080 to container 80, mounting a volume
docker run -d --name warmup -p 8080:80 -v ~/vg-warmup:/usr/share/nginx/html:ro nginx:alpine
docker ps
curl -s localhost:8080                       # served from the mounted folder

: STEP 2 logs
docker logs warmup | tail -5

: STEP 3 shell inside the container
docker exec warmup sh -c 'whoami; hostname; ls /usr/share/nginx/html; cat /etc/os-release | head -2'

: STEP 4 list, stop, remove containers and images
docker ps -a
docker stop warmup && docker rm warmup
docker images nginx
docker rmi nginx:alpine
docker ps -a
