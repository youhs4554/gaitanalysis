#!/bin/bash
docker stop nginx-container
docker rm nginx-container
docker build -t nginx-fileserver . 
docker run --name nginx-container -p 8080:80 -v /media/hossay/hdd1/:/usr/share/nginx/html:ro -d nginx-fileserver
