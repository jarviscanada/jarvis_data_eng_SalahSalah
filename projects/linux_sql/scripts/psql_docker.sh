#! /bin/bash

# script "[Start/Stop/Create] [db_username] [db_password]"

# Setup Arguments
cmd=$1
db_username=$2
db_password=$3

# Initialize Docker with Status Message
echo -n "Docker Status: "
sudo systemctl is-active docker || sudo systemctl start docker

# check container status
docker container inspect jrvs-psql
container_status=$?

# Handle Operations
case $cmd in
  create)
    # Check if Container already exists
    if [ $container_status -eq 0 ]; then
      echo "Container Already Exists"
      exit 1
    fi

    # Check Valid CLI Arguments
    if [ $# -ne 3 ]; then
      echo "Requires 2 additional arguments 'username' & 'password'"
      exit 1
    fi

    # Create Container
    docker volume create pgdata
    # Start Container
    docker run --name jrvs-psql -e POSTGRES_PASSWORD="password"  -d -v pgdata:/var/lib/postgresql/data -p 5432:5432 postgres:9.6-alpine
    # Exit and display if error
    exit $?
    ;;

  start|stop)
    if [ $container_status -ne 0 ]; then
      echo "Container has not been created please create container"
      exit 1
    fi

    docker container "$cmd" jrvs-psql
    exit $?
    ;;
  *)
    echo "Illegal Command"
    echo "Valid Commands: start|stop|create"
    exit 1
    ;;
esac