#!/bin/bash
set -e
TAG=iloveairflow
EAI_GT_PATH=`pwd`

docker build -t $TAG .

docker run \
       -v $EAI_GT_PATH:/eai_rsp_gt \
       -v $EAI_GT_PATH/eai_graph_tools/airflow_data/dags:/usr/local/airflow/dags \
       -p 8080:8080 \
       $TAG \
       /eai_rsp_gt/docker_launch_scripts/launch_webserver.sh
