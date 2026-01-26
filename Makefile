SHELL=/bin/bash
DATETIME:=$(shell date -u +%Y%m%dT%H%M%SZ)

### This is the Terraform-generated header for timdex-embeddings-dev. If  ###
###   this is a Lambda repo, uncomment the FUNCTION line below            ###
###   and review the other commented lines in the document.               ###
ECR_NAME_DEV := timdex-embeddings-dev
ECR_URL_DEV := 222053980223.dkr.ecr.us-east-1.amazonaws.com/timdex-embeddings-dev
### End of Terraform-generated header                                     ###

help: # Preview Makefile commands
	@awk 'BEGIN { FS = ":.*#"; print "Usage:  make <target>\n\nTargets:" } \
/^[-_[:alpha:]]+:.?*#/ { printf "  %-15s%s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ensure OS binaries aren't called if naming conflict with Make recipes
.PHONY: help venv install update test coveralls lint black mypy ruff safety lint-apply black-apply ruff-apply check-arch dist-dev publish-dev docker-clean

##############################################
# Python Environment and Dependency commands
##############################################

install: .venv .git/hooks/pre-commit # Install Python dependencies and create virtual environment if not exists
	uv sync --group dev --group local

.venv: # Creates virtual environment if not found
	@echo "Creating virtual environment at .venv..."
	uv venv .venv

.git/hooks/pre-commit: # Sets up pre-commit hook if not setup
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install

venv: .venv # Create the Python virtual environment

update: # Update Python dependencies
	uv lock --upgrade
	uv sync --group dev --group local

######################
# Unit test commands
######################

test: # Run tests and print a coverage report
	uv run coverage run --source=embeddings -m pytest -vv
	uv run coverage report -m

coveralls: test # Write coverage data to an LCOV report
	uv run coverage lcov -o ./coverage/lcov.info

####################################
# Code quality and safety commands
####################################

lint: black mypy ruff safety # Run linters

black: # Run 'black' linter and print a preview of suggested changes
	uv run black --check --diff .

mypy: # Run 'mypy' linter
	uv run mypy .

ruff: # Run 'ruff' linter and print a preview of errors
	uv run ruff check .

safety: # Check for security vulnerabilities
	uv run pip-audit --ignore-vuln CVE-2025-2953 --ignore-vuln CVE-2025-3730

lint-apply: black-apply ruff-apply # Apply changes with 'black' and resolve 'fixable errors' with 'ruff'

black-apply: # Apply changes with 'black'
	uv run black .

ruff-apply: # Resolve 'fixable errors' with 'ruff'
	uv run ruff check --fix .


####################################
# Developer Build and Deploy Commands for Dev environment in AWS
####################################

# Capture the SHA of the latest local developer commit on the feature branch
GIT_SHA := $(shell git describe --always)

# For validation testing of the .aws-architecture file
VALID_ARCH := linux/amd64 linux/arm64

# Extract/set the architecture for GPU builds and non-GPU builds from the
# .aws-architecture file, defaulting to "linux/amd64" if the key does not
# exist in the file.
GPU_ARCH := $(shell jq -r '.gpu // "linux/amd64"' .aws-architecture 2>/dev/null)
GPU_TAG := $(shell echo $(GPU_ARCH) | cut -d'/' -f2)-gpu
CPU_ARCH := $(shell jq -r '.cpu // "linux/amd64"' .aws-architecture 2>/dev/null)
CPU_TAG := $(shell echo $(CPU_ARCH) | cut -d'/' -f2)-cpu

validate-arch: ## Ensure that the parsing of the .aws-architecture file provided valid values
	@if [ ! -f .aws-architecture ]; then \
		echo "WARN: .aws-architecture not found. Using defaults gpu=linux/amd64, cpu=linux/amd64"; \
	fi
	@for value in $(GPU_ARCH) $(CPU_ARCH); do \
		case " $(VALID_ARCH) " in \
		*" $$value "*) ;; \
		*) echo "ERROR: Invalid architecture: $$value" >&2; exit 1;; \
		esac; \
	done
	@echo "Validation passed: gpu=$(GPU_ARCH), cpu=$(CPU_ARCH)"

ensure-builder: ## Ensures the the buildx builder is ready to go
	@echo "Prepare the Docker BuildX builder"; \
	docker buildx inspect $(ECR_NAME_DEV) >/dev/null 2>&1 || docker buildx create --name $(ECR_NAME_DEV) --driver docker-container --use; \
	docker buildx use $(ECR_NAME_DEV); \
	docker buildx inspect --bootstrap >/dev/null; \
	docker buildx prune -af --filter until=24h || true; \
	echo "BuildX Builder Ready!"

dist-dev-gpu: validate-arch ensure-builder ## Build GPU-enabled docker container (intended for developer-based manual build)
	@echo "Build GPU-enabled container (for $(GPU_ARCH))"
	@docker buildx build --platform $(GPU_ARCH) \
		--file Dockerfile-gpu \
		--progress=plain \
		--load \
	    --tag $(ECR_URL_DEV):latest-$(GPU_TAG) \
		--tag $(ECR_URL_DEV):make-$(GIT_SHA)-$(GPU_TAG) \
		--tag $(ECR_NAME_DEV):latest-$(GPU_TAG) \
		.
	@echo "Build for GPU-enabled container is done!"

dist-dev-cpu: validate-arch ensure-builder ## Build non-GPU docker container (intended for developer-based manual build)
	@echo "Build CPU container (for $(CPU_ARCH))"
	@docker buildx build --platform $(CPU_ARCH) \
		--file Dockerfile-cpu \
		--progress=plain \
		--load \
	    --tag $(ECR_URL_DEV):latest-$(CPU_TAG) \
		--tag $(ECR_URL_DEV):make-$(GIT_SHA)-$(CPU_TAG) \
		--tag $(ECR_NAME_DEV):latest-$(CPU_TAG) \
		.
	@echo "Build for CPU container is done!"

dist-dev-all: dist-dev-gpu dist-dev-cpu ## Runs both the GPU and the CPU builds

publish-dev-gpu: dist-dev-gpu ## Build, tag and push GPU-enabled container (intended for developer-based manual publish)
	@aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(ECR_URL_DEV); \
	docker push $(ECR_URL_DEV):latest-$(GPU_TAG); \
	docker push $(ECR_URL_DEV):make-$(GIT_SHA)-$(GPU_TAG)
	@echo "Cleaning up dangling Docker images..."; \
	docker image prune -f --filter "dangling=true"

publish-dev-cpu: dist-dev-cpu ## Build, tag and push no-GPU container (intended for developer-based manual publish)
	@aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(ECR_URL_DEV); \
	docker push $(ECR_URL_DEV):latest-$(CPU_TAG); \
	docker push $(ECR_URL_DEV):make-$(GIT_SHA)-$(CPU_TAG)
	@echo "Cleaning up dangling Docker images..."; \
	docker image prune -f --filter "dangling=true"

publish-dev-all: publish-dev-gpu publish-dev-cpu ## Publish both images to AWS

docker-clean: ## Clean up Docker detritus
	echo "Cleaning up Docker leftovers (containers, images, builders)"; \
	docker rmi -f $(ECR_URL_DEV):latest-$(GPU_TAG) $(ECR_URL_DEV):latest-$(CPU_TAG) || true; \
	docker rmi -f $(ECR_URL_DEV):make-$(GIT_SHA)-$(GPU_TAG) $(ECR_URL_DEV):make-$(GIT_SHA)-$(CPU_TAG) || true; \
    docker rmi -f $(ECR_NAME_DEV):latest-$(GPU_TAG) $(ECR_NAME_DEV):latest-$(CPU_TAG) || true; \
	docker buildx rm $(ECR_NAME_DEV) || true; \
	docker buildx prune -af || true
