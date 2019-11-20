#!/usr/bin/env bash
set -Eeo pipefail

# Stripped down version of https://github.com/docker-library/postgres/blob/ff832cbf1e9ffe150f66f00a0837d5b59083fec9/10/docker-entrypoint.sh
# This script manages initializing the database to run as an underprivileged user
# It also has the nice property of leaving an existing initialization untouched, so data can persist across runs

file_env() {
	local var="$1"
	local fileVar="${var}_FILE"
	local def="${2:-}"
	if [ "${!var:-}" ] && [ "${!fileVar:-}" ]; then
		echo >&2 "error: both $var and $fileVar are set (but are exclusive)"
		exit 1
	fi
	local val="$def"
	if [ "${!var:-}" ]; then
		val="${!var}"
	elif [ "${!fileVar:-}" ]; then
		val="$(< "${!fileVar}")"
	fi
	export "$var"="$val"
	unset "$fileVar"
}

start_server() {
    pg_ctl -D $PGDATA -l $PGLOGS start
}

mkdir -p "$PGDATA"
chown -R "$(id -u)" "$PGDATA" 2>/dev/null || :
chmod 700 "$PGDATA" 2>/dev/null || :
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    # "initdb" is particular about the current user existing in "/etc/passwd", so we use "nss_wrapper" to fake that if necessary
    # see https://github.com/docker-library/postgres/pull/253, https://github.com/docker-library/postgres/issues/359, https://cwrap.org/nss_wrapper.html
    if ! getent passwd "$(id -u)" &> /dev/null && [ -e /usr/lib/libnss_wrapper.so ]; then
	export LD_PRELOAD='/usr/lib/libnss_wrapper.so'
	export NSS_WRAPPER_PASSWD="$(mktemp)"
	export NSS_WRAPPER_GROUP="$(mktemp)"
	echo "postgres:x:$(id -u):$(id -g):PostgreSQL:$PGDATA:/bin/false" > "$NSS_WRAPPER_PASSWD"
	echo "postgres:x:$(id -g):" > "$NSS_WRAPPER_GROUP"
    fi

    file_env 'POSTGRES_USER' 'postgres'
    file_env 'POSTGRES_PASSWORD'

    file_env 'POSTGRES_INITDB_ARGS'
    if [ "$POSTGRES_INITDB_WALDIR" ]; then
	export POSTGRES_INITDB_ARGS="$POSTGRES_INITDB_ARGS --waldir $POSTGRES_INITDB_WALDIR"
    fi
    eval 'initdb --username="$POSTGRES_USER" --pwfile=<(echo "$POSTGRES_PASSWORD") '"$POSTGRES_INITDB_ARGS"

    # unset/cleanup "nss_wrapper" bits
    if [ "${LD_PRELOAD:-}" = '/usr/lib/libnss_wrapper.so' ]; then
	rm -f "$NSS_WRAPPER_PASSWD" "$NSS_WRAPPER_GROUP"
	unset LD_PRELOAD NSS_WRAPPER_PASSWD NSS_WRAPPER_GROUP
    fi

    start_server

    # Create the database for Airflow
    psql -U postgres -c "CREATE DATABASE airflow"
else
    start_server
fi


