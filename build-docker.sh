#!/bin/bash
set -e

VERSION=$1

# fixme check for version
if [ -z "$1" ] ; then
    echo "Usage: build-docker.sh VERSION"
    exit 1
fi

echo "Building version $1"

docker build -f Dockerfile -t ebreviainc/kirke:$VERSION .
docker push ebreviainc/kirke:$VERSION

# tag the build in git
git tag -f $VERSION
git push -f origin $VERSION

# generate aws deploy file
# FIXME should do this for each region, since they need separate deploy files (stupid)
# for us, lon, and ca
mkdir -p target/aws
cp -r docker/.ebextensions target/aws

for REGION in "us" "lon" "ca"; do
    cp docker/Dockerrun.aws.json target/aws/Dockerrun.aws.json
    sed -i='' "s/<VERSION>/$VERSION/" target/aws/Dockerrun.aws.json
    sed -i='' "s/<REGION>/$REGION/" target/aws/Dockerrun.aws.json
    (cd target/aws && zip -r kirke-$REGION-$VERSION.zip Dockerrun.aws.json .ebextensions)
    rm target/aws/Dockerrun*
done

echo "Build complete! AWS deployment files in target/aws/"
