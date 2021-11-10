
docker build -t m400 .

docker run --rm -it -p 8866:8866 --name m400test --mount type=bind,source="$(pwd)"/data,target=/datamount m400:latest
