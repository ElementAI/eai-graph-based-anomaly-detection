#!/bin/bash
set -e

# Set up the conda environment
source /conda3/setup.sh
source $AIRFLOW_USER_HOME/setup_fernet_key.sh

# Supports a common setup where the user mounts /usr/local/airflow/dags directly, which makes cp error
if [ -z "$(ls /usr/local/airflow/dags/*.py)" ];
then
    # Bring DAGs into the location that Airflow expects
    cp $WD/eai_graph_tools/airflow_data/dags/*.py /usr/local/airflow/dags
fi

# Set up the PostgreSQL environment
$WD/docker_launch_scripts/pg-docker-entrypoint.sh

export DASK_HOST=`hostname`
export DASK_PORT="8786"
# Runtime configuration so that the correct address can be used
# https://airflow.readthedocs.io/en/1.9.0/configuration.html
export AIRFLOW__DASK__CLUSTER_ADDRESS="$DASK_HOST:$DASK_PORT"

# Start the Dask scheduler, so it listens on all addresses.
dask-scheduler --host 0.0.0.0 --port $DASK_PORT &

echo "Launching Dask workers"
. ./docker_launch_scripts/run_dask_workers.sh

# Initialize Airflow database backend
airflow initdb

# Start Airflow
airflow scheduler &
airflow webserver -p 8080
