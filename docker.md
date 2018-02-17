# build
docker build [OPTIONS] PATH | URL | -
docker build -t tag -f Dockerfile context

# build and remove intermediary images during build process
docker build --rm -t tag -f Dockerfile context

# run (assign name, interactive, removed when exit)
docker run [OPTIONS] image [COMMAND] [ARG...]
docker run -rm --name test -v /path:/path-inside-container -ti image

# run container and give it a name (detached)
docker run -d -name tmp image

# attach interactive
docker start -ai container-id

# find id of last run container
docker ps -l -q

# commit
docker commit $(docker ps -l -q) tmp:version3

# inspect
docker inspect $(docker ps -l -q) | jq -r '.[0].NetworkSettings.IPAddress'

# list all exited containers
docker ps -aq -f status=exited

# remove containers (runnings ones will fail and not be removed)
docker rm $(docker ps -aq --no-trunc)

# remove dangling/untagged images
docker rmi $(docker images -q --filter dangling=true)

# use --rm together with docker build to remove intermediary images during build process
docker build --rm ...

# cleanup
  - docker system df
  - docker system prune
  - docker volume prune
  - docker network prune
  - docker container prune
  - docker image prune

# reference
https://docs.docker.com/engine/reference/commandline/run/
