#!/bin/sh
IMAGE=eai-graph-tools
pipenv lock &&
jq --raw-output \
    '.default | to_entries[] | select(.value.editable | not) | "\(.key)\(.value.version)"' \
    Pipfile.lock \
    >requirements.txt &&
docker build --tag $IMAGE . || exit $?

if [ -f .tag_map ]; then
    cat .tag_map | tr -s '\n' | awk '{printf("docker build --tag %s --file %s . && docker push %s\n", $1, $2, $1)}' | sh
else
    echo
    echo "Tag map not found. Create a file named .tag_map in this directory, where"
    echo "each line follows the format"
    echo
    echo "<tag> <dockerfile.name>"
    echo
    echo "and the image derived from the named dockerfile will be built, tagged"
    echo "with the given tag, and then pushed. Any number of blanks can appear"
    echo "between the tag and the file name. For example, this .tag_map:"
    echo
    echo "my-user-name:v1                     Dockerfile.mine"
    echo "remote-repo.net/image-name:version  Dockerfile.remote"
    echo
    echo "will build two images after the base image, and tag and push them"
    echo "respectively to the default Docker repo, and to the private Docker"
    echo "repo at address remote-repo.net. Both these dockerfiles can start"
    echo "with statement FROM $IMAGE."
fi
