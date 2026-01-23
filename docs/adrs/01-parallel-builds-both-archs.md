# 1. Parallel Builds for both ARM64 and AMD64

## Status

Proposed

## Context

This application was originally conceived as an ECS Fargate Task, but times have changed. It is now very likely that this app will be run through AWS Batch, sometimes using GPU-enabled host servers and sometimes running as a simple Fargate task.

In AWS Batch, GPU-enabled host machines only run on the AMD64 hardware (with NVIDIA GPUs attached). ARM64 (e.g., Graviton) is **not** supported for GPU-enabled EC2 instances. See [Batch EC2 Configuration](https://docs.aws.amazon.com/batch/latest/APIReference/API_Ec2Configuration.html) and [ECS-Optimized AMIs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-optimized_AMI.html#gpuami). For the Fargate execution in AWS Batch, the ARM64 hardware is more efficient and less expensive.

Our current CD workflows are built for building on either one CPU architecture or the other (`amd64` or `arm64`), but not for both. This will be the first use case for an application that will need separate and parallel builds for both AMD64 and ARM64. We could build just one multi-architecture image, but that would make for a very large image. Additionally, there will likely be different Dockerfiles for each CPU architecture based on how the container should be built for GPU-enabled jobs versus jobs with no GPU.

At this time, it looks like this is the only repository that will need this special treatment for builds, so instead of trying to build a shared workflow in our [.github]() repository, we will update the local workflows to do all the work. If, in the future, we need to do something similar for another repository, we can decide to move this into a shared CD workflow. For now, it's okay to keep this as a one-off workflow here.

## Decision

Rebuild the dev, stage, and prod GHA workflows to build both AMD64 and ARM64 containers and push both container images to the ECR Repository in AWS. Additionally, update the `Makefile` commands for `dev` to make it easy to build one or both container images for a developer-based push to Dev1.
