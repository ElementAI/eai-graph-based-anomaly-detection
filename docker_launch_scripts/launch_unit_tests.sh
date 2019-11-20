#!/bin/bash
source /conda3/setup.sh
source $AIRFLOW_USER_HOME/setup_fernet_key.sh

# Runtime configuration to override the PostgreSQL set up and use simpler SQLite
# https://airflow.readthedocs.io/en/1.9.0/configuration.html
export AIRFLOW__CORE__SQL_ALCHEMY_CONN="sqlite:////tmp/airflow.db"
export AIRFLOW__CORE__EXECUTOR="SequentialExecutor"

# Initialize Airflow database backend
airflow initdb

exec ./runtests.py

# The exec statement should end this process; if we make it to here, something
# was wrong with the presence of the check script.
cat <<EOF 1>&2
Script runtests.py not present in current directory. Fail.
EOF
exit 1
