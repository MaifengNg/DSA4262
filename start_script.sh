#!/bin/sh -v

echo 'building docker image'
docker build -f Dockerfile -t rnaclassifier .
wait

echo 'running rnaclassifiercontainer container'
docker run -dit --name rnaclassifiercontainer rnaclassifier
docker logs --follow rnaclassifiercontainer
wait

echo 'copying from docker to local'
docker start rnaclassifiercontainer
docker cp rnaclassifiercontainer:/usr/src/app/Results/. ./Results
wait

echo 'killing container'
docker kill rnaclassifiercontainer
wait

echo 'remove container'
docker rm rnaclassifiercontainer