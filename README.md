# eai-graph-based-anomaly-detection

This project implements experiments for doing anomaly detection on node embeddings. Experiments are implemented using 
Airflow, which runs from a Docker image. 


### Launch Airflow webserver from Docker

Build and run the docker container:
```
$ docker build -t eai_gt:v1 .
$ docker run -it -p 8080:8080 --mount type=bind,source=$EAI_GT_PATH/eai_graph_tools/,target=/eai_rsp_gt/eai_graph_tools --mount type=bind,source=$EAI_GT_PATH/dataset_files/,target=/eai_rsp_gt/dataset_files eai_gt:v1 bash
```

From the container, launch the webserver:
```
user@0cc9e8e656b5:/eai_rsp_gt$ ./docker_launch_scripts/launch_webserver.sh 
```

Connect to the airflow webserver: http://localhost:8080
Note: dags used by airflow are located at "/usr/local/airflow/dags" on the container.

### Running the unit tests locally
```
$ docker build --build-arg UID=$(id -u) --tag test .
$ docker run test
```

### Multi-stage Docker image

The Docker image is built so it may be deployed on a compute infrastructure
where the container running the image is not itself run as root; the container
will work if run as an unprivileged user. In order to effectively run locally
without undue root privileges, the Docker image is built over two
[stages](https://docs.docker.com/develop/develop-images/multistage-build/):

1. Base image (named `base`). If tagged, the container will run as whatever
   user is set by the underlying container orchestration platform. Locally, it
   will run as root.
1. Image altered to run locally without root privileges. The container started
   from this stage (named `local`) will run as unprivileged user named `user`,
   with UID given as build argument. It is wise to use the current host user's
   UID, particularly if one bind-mounts their code directories to
   subdirectories of `/eai_rsp_gt` on the container: this will make
   byte-compiled Python directories (`__pycache__`) belong to the host user,
   enabling subsequent `docker build` statements without eliminating all
   `__pycache__` subdirectories.

To clarify, for running locally, one should usually build the image with the
UID build argument:

```
docker build --build-arg UID=$(id -u) --tag eai-graph-tools-local .
```

This clears off any problem inherent to Python bytecode caching when
bind-mounting the code. The computations on the image when it is `docker run`
will be executed as unprivileged user `user`, with one's own UID.

For running on a container orchestration platform that sets, by itself, the
user as which runs the container, merely build and push the base image
instead.

```
docker build --target base --tag remote.orchestration/platform/url:version .
docker push remote.orchestration/platform/url:version
```
