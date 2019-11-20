FROM nvidia/cuda:latest AS base

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux
ENV AIRFLOW_USER_HOME=/usr/local/airflow
ENV AIRFLOW_HOME=/usr/local/airflow

# "Cache busting" a la https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
RUN apt-get update && apt-get install -y \
        build-essential \
        apt-utils \
        curl \
        rsync \
        netcat \
        locales \
        vim-nox \
        emacs-nox \
        less \
        htop

# Download and install Anaconda/Miniconda, and put its tool setup into a
# sourceable script.
ENV URL_ANACONDA=https://repo.anaconda.com/miniconda/Miniconda2-4.7.10-Linux-x86_64.sh
RUN curl -o install.sh $URL_ANACONDA
RUN sh install.sh -b -p /conda3
RUN rm install.sh
RUN /conda3/bin/conda update -n base -c defaults conda
RUN /conda3/bin/conda shell.bash hook >/conda3/setup.sh
RUN chmod a+r /conda3/setup.sh
COPY condarc /conda3/.condarc

# Anaconda does not work well with basic shells, so let's do all our setup
# onwards with bash
SHELL ["bash", "-c"]

# We set up Python tools in a non-base Conda environment.
RUN source /conda3/setup.sh \
    && conda create -n env python=3.6 numpy \
    && echo 'conda activate env' >>/conda3/setup.sh

# Airflow and dependencies.
RUN apt-get update && apt-get install -y\
    freetds-dev \
    libkrb5-dev \
    libsasl2-dev \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    freetds-bin \
    libmysqlclient-dev \
    postgresql-client \
    postgresql

# Set up locales and use en_US. Copied from puckel/docker-airflow
RUN sed -i 's/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8

# Install Airflow using pip. We first install numpy through conda, because on
# PyPI, numpy has a later version, which conda otherwise clobbers when
# installing the deps to our own code.
COPY airflow_requirements.txt .
RUN source /conda3/setup.sh && \
    pip install --upgrade pip && \
    pip install -r airflow_requirements.txt && \
    rm airflow_requirements.txt

# Install our own code dependencies.
COPY requirements.txt .
RUN source /conda3/setup.sh && \
    conda install --file requirements.txt && rm requirements.txt && \
    pip install \
        torch-geometric \
        torch-cluster \
        torch-scatter \
        torch-sparse

ENV WD /eai_rsp_gt
WORKDIR $WD

# Set up PostgreSQL
ENV PGDATA $WD/pgdata
RUN mkdir $PGDATA
ENV PGLOGS $PGDATA/pglogs

# Make everything world-accessible, for underprivileged users
RUN chmod -R a+rwX .
# Write privileges are required here for the server to manage locking without running as root
RUN chmod a+w /var/run/postgresql

# Simplify accessing the postgres scripts
ENV PATH "$PATH:/usr/lib/postgresql/10/bin"


# Deploy Airflow configuration and dags, moving the Airflow plug-in to the
# proper location.
# - Note: the dag directory is open to the world so that the underprivileged
#   user can get in.
# - TODO: there must be a better way
RUN mkdir -p $AIRFLOW_USER_HOME/dags && \
    touch $AIRFLOW_USER_HOME/unittests.cfg
COPY airflow_dock/airflow.cfg ${AIRFLOW_USER_HOME}/airflow.cfg
COPY airflow_plugins /usr/local/airflow

# Generate Fernet key.
COPY generate_fernet_key.py .
RUN source /conda3/setup.sh && \
    echo "export FERNET_KEY=$(python generate_fernet_key.py)" >$AIRFLOW_USER_HOME/setup_fernet_key.sh && \
    rm generate_fernet_key.py

# Initialize the Airflow database.
RUN source /conda3/setup.sh && \
    source $AIRFLOW_USER_HOME/setup_fernet_key.sh && \
    chmod -R a+rwX $AIRFLOW_USER_HOME

# Get code.
COPY eai_graph_tools eai_graph_tools
COPY setup.py .
COPY README.md .

# Mark the code as an editable module, but arranged so that it is set up
# outside of root's home directory -- this one gets completely locked down on
# eAI's compute infra. The PATH mod resolves a warning when pip-installing
# with --user.
ENV PYTHONUSERBASE=/usr/local
ENV PATH="$PATH:$PYTHONUSERBASE/bin"
RUN source /conda3/setup.sh && \
    mkdir -p $PYTHONUSERBASE/lib/python3.6/site-packages && \
    pip install --user -e . && \
    chmod -R a+rX $PYTHONUSERBASE

# Get launch scripts.
COPY docker_launch_scripts docker_launch_scripts

# Get unit tests.
COPY tests tests
COPY .flake8 .
COPY runtests.py .

# Default computation is to run the Airflow server.
EXPOSE 8080
CMD $WD/docker_launch_scripts/launch_unit_tests.sh


FROM base AS local

# Make unprivileged user and compute from it.
ARG UID
RUN adduser --disabled-login --gecos '' --uid $UID user
RUN chown -R user:user $WD
USER user

# Automate Anaconda and Airflow configurations when running an interactive
# shell.
RUN echo "source /conda3/setup.sh; source /usr/local/airflow/setup_fernet_key.sh" >>/home/user/.bashrc
