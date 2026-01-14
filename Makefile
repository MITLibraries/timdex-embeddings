SHELL=/bin/bash
DATETIME:=$(shell date -u +%Y%m%dT%H%M%SZ)
VALID_ARCH := linux/amd64 linux/arm64
GPU_ARCH := $(shell jq -r '.gpu // "linux/amd64"' .aws-architecture 2>/dev/null)
NOGPU_ARCH := $(shell jq -r '.nogpu // "linux/amd64"' .aws-architecture 2>/dev/null)

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
.PHONY: help venv install update test coveralls lint black mypy ruff safety lint-apply black-apply ruff-apply \
	validate-arch dist-dev-gpu dist-dev-nogpu dist-dev publish-dev docker-clean

##############################################
# Python Environment and Dependency commands
##############################################

install: .venv .git/hooks/pre-commit # Install Python dependencies and create virtual environment if not exists
	uv sync --dev

.venv: # Creates virtual environment if not found
	@echo "Creating virtual environment at .venv..."
	uv venv .venv

.git/hooks/pre-commit: # Sets up pre-commit hook if not setup
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install

venv: .venv # Create the Python virtual environment

update: # Update Python dependencies
	uv lock --upgrade
	uv sync --dev

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
	uv run pip-audit

lint-apply: black-apply ruff-apply # Apply changes with 'black' and resolve 'fixable errors' with 'ruff'

black-apply: # Apply changes with 'black'
	uv run black .

ruff-apply: # Resolve 'fixable errors' with 'ruff'
	uv run ruff check --fix .


####################################
# Docker
####################################

docker-build: # Build local image for testing
	docker build --platform $(CPU_ARCH) -t timdex-embeddings:latest .

docker-shell: # Shell into local container for testing
	docker run -it --entrypoint='bash' timdex-embeddings:latest

docker-run: # Run main entrypoint + command without arguments
	docker run timdex-embeddings:latest

####################################
# Developer Build and Deploy Commands for Dev environment in AWS
####################################

validate-arch: ## Ensure that the .aws-architecture file provided valid values
	@for value in $(GPU_ARCH) $(NOGPU_ARCH); do \
		case " $(VALID_ARCH) " in \
		*" $$value "*) ;; \
		*) echo "ERROR: Invalid architecture: $$value" >&2; exit 1;; \
        esac; \
    done; \
    echo "Validation passed: gpu=$(GPU_ARCH), nogpu=$(NOGPU_ARCH)"

ensure-builder: ## Ensures the the buildx builder is ready to go
	@echo "Prepare the Docker BuildX builder"; \
	docker buildx inspect $(ECR_NAME_DEV) >/dev/null 2>&1 || docker buildx create --name $(ECR_NAME_DEV) --driver docker-container --use; \
	docker buildx use $(ECR_NAME_DEV); \
	docker buildx inspect --bootstrap >/dev/null; \
	docker buildx prune -af --filter until=24h || true; \

dist-dev-gpu: validate-arch ensure-builder ## Build GPU-enabled docker container (intended for developer-based manual build)
	@ARCH_TAG="latest-$(shell echo $(GPU_ARCH) | cut -d'/' -f2)"; \
	echo "Build GPU-enabled container (for $(GPU_ARCH))"; \
	docker buildx build --platform $(GPU_ARCH) \
		--file Dockerfile.gpu \
		--load \
	    --tag $(ECR_URL_DEV):$$ARCH_TAG \
	    --tag $(ECR_URL_DEV):make-$$ARCH_TAG \
		--tag $(ECR_URL_DEV):make-$(shell git describe --always)-$$ARCH_TAG \
		--tag $(ECR_NAME_DEV):$$ARCH_TAG \
		.

dist-dev-nogpu: validate-arch ensure-builder ## Build non-GPU docker container (intended for developer-based manual build)
	@ARCH_TAG="latest-$(shell echo $(NOGPU_ARCH) | cut -d'/' -f2)"; \
	echo "Build non-GPU container (for $(NOGPU_ARCH))"; \
	docker buildx build --platform $(NOGPU_ARCH) \
		--file Dockerfile.nogpu \
		--load \
	    --tag $(ECR_URL_DEV):$$ARCH_TAG \
	    --tag $(ECR_URL_DEV):make-$$ARCH_TAG \
		--tag $(ECR_URL_DEV):make-$(shell git describe --always)-$$ARCH_TAG \
		--tag $(ECR_NAME_DEV):$$ARCH_TAG \
		.

dist-dev: dist-dev-gpu dist-dev-nogpu ## Runs both the GPU and the NOGPU builds

publish-dev-gpu: ## Build, tag and push GPU-enabled container (intended for developer-based manual publish)
	@ARCH_TAG="latest-$(shell echo $(GPU_ARCH) | cut -d'/' -f2)"; \
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(ECR_URL_DEV); \
	docker push $(ECR_URL_DEV):$$ARCH_TAG; \
	docker push $(ECR_URL_DEV):make-$$ARCH_TAG; \
	docker push $(ECR_URL_DEV):make-$(shell git describe --always)-$$ARCH_TAG; \
    echo "Cleaning up dangling Docker images..."; \
    docker image prune -f --filter "dangling=true"

publish-dev-nogpu: ## Build, tag and push no-GPU conatiner (intended for developer-based manual publish)
	@ARCH_TAG="latest-$(shell echo $(NOGPU_ARCH) | cut -d'/' -f2)"; \
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(ECR_URL_DEV); \
	docker push $(ECR_URL_DEV):$$ARCH_TAG; \
	docker push $(ECR_URL_DEV):make-$$ARCH_TAG; \
	docker push $(ECR_URL_DEV):make-$(shell git describe --always)-$$ARCH_TAG; \
    echo "Cleaning up dangling Docker images..."; \
    docker image prune -f --filter "dangling=true"

publish-dev: publish-dev-gpu publish-dev-nogpu ## Publish both images to AWS

docker-clean: ## Clean up Docker detritus
	@GPUARCH_TAG="latest-$(shell echo $(GPU_ARCH) | cut -d'/' -f2)"; \
	NOGPUARCH_TAG="latest-$(shell echo $(NOGPU_ARCH) | cut -d'/' -f2)"; \
	echo "Cleaning up Docker leftovers (containers, images, builders)"; \
	docker rmi -f $(ECR_URL_DEV):$$GPUARCH_TAG $(ECR_URL_DEV):$$NOGPUARCH_TAG || true; \
	docker rmi -f $(ECR_URL_DEV):make-$$GPUARCH_TAG $(ECR_URL_DEV):make-$$NOGPUARCH_TAG || true; \
	docker rmi -f $(ECR_URL_DEV):make-$(shell git describe --always)-$$GPUARCH_TAG $(ECR_URL_DEV):make-$(shell git describe --always)-$$NOGPUARCH_TAG|| true; \
    docker rmi -f $(ECR_NAME_DEV):$$GPUARCH_TAG $(ECR_NAME_DEV):$$NOGPUARCH_TAG || true; \
	docker buildx rm $(ECR_NAME_DEV) || true; \
	docker buildx prune -af --filter until=24h || true
