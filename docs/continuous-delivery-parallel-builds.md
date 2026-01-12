# The CD pipeline for multiple, parallel builds

This application has a unique CD pipeline for pushing container images to AWS. All of our other applications will push either an AMD64-based image or an ARM64-based image to the ECR Repository. This application requires that there are both AMD64 **AND** ARM64 images available in the ECR Repository. Additionally, those different images will be built from different Dockerfiles!

## Two Dockerfiles

We separate the two builds by leveraging two different Dockerfiles:

* `Dockerfile.gpu`: The Dockerfile that defines the build for any GPU-enabled containers
* `Dockerfile.nogpu`: The Dockerfile that defines the build for containers that will not use GPUs

## CPU Architecture

We will stick with a single `.aws-architecture` file, but we will format it as a simple list of key/value pairs and leverage `jq` to parse the information in our `make` commands and GitHub workflows. The file will look like

```json
{
    "gpu": "linux/<arch>",
    "nogpu": "linux/<arch>"
}
```

Where `<arch>` is either `linux/amd64` or `linux/arm64`.

## Makefile configuration

The standard `Makefile` for our Python applications that can pick a single CPU Architecture has a `CPU_ARCH` variable defined that is just read from the `.aws_architecture` file. That won't work here. Now our `Makefile` will set both `GPU_ARCH` and `NOGPU_ARCH` variables and use those for all the targets.
