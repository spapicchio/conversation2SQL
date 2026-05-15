# HPC Singularity Deployment Design

**Date:** 2026-05-10
**Branch:** feat/baselines-ablation

## Goal

Run the conversation2SQL evaluation pipeline on an HPC cluster with no internet on compute nodes, using Singularity instead of Docker Compose.

## Constraints

- Compute nodes: no internet access, GPU available, SLURM scheduler
- Login node: internet access, but Singularity **not available** and insufficient disk space
- Local machine: internet access, Docker and Singularity available
- Shared filesystem (Lustre/GPFS/NFS): accessible from both login and compute nodes via `$SCRATCH`
- Singularity networking: CNI disabled for non-root users → containers inherit host network namespace (equivalent to `network_mode: host`)
- LLM backend: local VLLM (no outbound API calls from compute nodes)

## Approach

Three Singularity instances in one SLURM job — one compute container + two PostgreSQL containers — mirroring the Docker Compose topology without any code changes.

## Phase 1: Local machine (one-time)

### 1.1 Build SIF files

```bash
# Option A: Singularity available locally
export SINGULARITY_CACHEDIR=/tmp/singularity_cache
singularity pull verl.sif        docker://verlai/verl:vllm017.latest
singularity pull pg_lite.sif     docker://shawnxxh/bird-interact-postgresql:latest
singularity pull pg_full.sif     docker://shawnxxh/bird-interact-postgresql-full:latest

# Option B: Docker only locally
docker pull verlai/verl:vllm017.latest
singularity build verl.sif        docker-daemon://verlai/verl:vllm017.latest
singularity build pg_lite.sif     docker-daemon://shawnxxh/bird-interact-postgresql:latest
singularity build pg_full.sif     docker-daemon://shawnxxh/bird-interact-postgresql-full:latest
```

### 1.2 Patch PostgreSQL SIFs for HPC UID mapping (if needed)

PostgreSQL images run their data directory as the `postgres` OS user (UID 70 or 999). Singularity runs as the HPC user UID, which cannot write to that directory. If the instances fail to start with permission errors, rebuild with relaxed permissions:

```dockerfile
# Dockerfile.pg-hpc-lite
FROM shawnxxh/bird-interact-postgresql:latest
RUN chmod -R 777 /var/lib/postgresql/data
```

```dockerfile
# Dockerfile.pg-hpc-full
FROM shawnxxh/bird-interact-postgresql-full:latest
RUN chmod -R 777 /var/lib/postgresql/data
```

```bash
# Build patched images and convert to SIF
docker build -f Dockerfile.pg-hpc-lite -t pg-lite-hpc .
docker build -f Dockerfile.pg-hpc-full -t pg-full-hpc .

singularity build pg_lite.sif  docker-daemon://pg-lite-hpc
singularity build pg_full.sif  docker-daemon://pg-full-hpc
```

### 1.3 Download model weights

```bash
HF_HOME=./hf_cache huggingface-cli download <model-id>
```

### 1.4 Transfer to HPC

```bash
# Create target directories
ssh <user>@<hpc-login> "mkdir -p \$SCRATCH/sif \$SCRATCH/hf_cache"

# Transfer SIF files, repo, and model weights
rsync -avP verl.sif pg_lite.sif pg_full.sif <user>@<hpc-login>:\$SCRATCH/sif/
rsync -avP ./hf_cache/                       <user>@<hpc-login>:\$SCRATCH/hf_cache/
rsync -avP /path/to/conversation2SQL/        <user>@<hpc-login>:\$SCRATCH/conversation2SQL/
```

## Phase 2: HPC first-time setup (one-off interactive job)

Bootstrap the Python venv inside the compute container. This runs once; subsequent jobs skip it via `uv sync`'s no-op behaviour when the lockfile is unchanged.

```bash
srun --pty --time=00:30:00 --mem=16G bash
module load singularity

singularity exec \
  --bind $SCRATCH/conversation2SQL:/workspaces/conversation2SQL \
  $SCRATCH/sif/verl.sif bash -c "
    cd /workspaces/conversation2SQL
    uv venv --system-site-packages
    uv sync
  "
```

## Phase 3: SLURM job script (`slurm/run.sh`)

```
$SCRATCH/
  sif/
    verl.sif
    pg_lite.sif
    pg_full.sif
  conversation2SQL/       ← repo clone (with .venv inside)
  hf_cache/               ← model weights
```

```bash
#!/bin/bash
#SBATCH --job-name=conv2sql
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

module load singularity

SIF=$SCRATCH/sif
REPO=$SCRATCH/conversation2SQL
HF=$SCRATCH/hf_cache
ID=${SLURM_JOB_ID}

# Cleanup on exit (including SLURM time-limit SIGTERM)
trap "singularity instance stop pg_lite_${ID} 2>/dev/null; \
      singularity instance stop pg_full_${ID} 2>/dev/null" EXIT

# --- 0. Bootstrap venv (fast no-op if lockfile unchanged) ---
singularity exec \
  --bind $REPO:/workspaces/conversation2SQL \
  $SIF/verl.sif bash -c "
    cd /workspaces/conversation2SQL
    uv venv --system-site-packages
    uv sync
  "

# --- 1. Start PostgreSQL instances ---
# pg_lite listens on default 5432
singularity instance start --writable-tmpfs \
  $SIF/pg_lite.sif pg_lite_${ID} \
  -c max_connections=300 -c shared_buffers=256MB

# pg_full must listen on 5433 to avoid collision on the shared host network
singularity instance start --writable-tmpfs \
  $SIF/pg_full.sif pg_full_${ID} \
  -c port=5433 -c max_connections=300 -c shared_buffers=256MB

# --- 2. Wait for both to accept connections ---
echo "Waiting for PostgreSQL lite (5432)..."
until nc -z localhost 5432 2>/dev/null; do sleep 2; done
echo "Waiting for PostgreSQL full (5433)..."
until nc -z localhost 5433 2>/dev/null; do sleep 2; done
echo "PostgreSQL ready."

# --- 3. Run experiment (pass extra flags via sbatch args) ---
singularity exec --nv \
  --bind $REPO:/workspaces/conversation2SQL \
  --bind $HF:/hf_cache \
  --env HF_HOME=/hf_cache \
  $SIF/verl.sif \
  bash -c 'cd /workspaces/conversation2SQL && source .venv/bin/activate && python main.py --config configs/eval_pipeline_config.yaml "$@"' \
  -- "$@"
```

Submit:
```bash
sbatch slurm/run.sh                                        # default config
sbatch slurm/run.sh --baseline no_tool --debug true        # override flags
```

## Key design decisions

| Decision | Rationale |
|---|---|
| `--writable-tmpfs` for PostgreSQL | BIRD data is baked into the images; only runtime writes (WAL, PID, locks) need a writable overlay |
| `-c port=5433` passed to `pg_full` | Singularity host-network means no Docker port mapping; second PG instance must be told to use 5433 explicitly |
| `$SLURM_JOB_ID` in instance names | Prevents name collisions when multiple jobs run in parallel on the same node |
| `trap EXIT` for cleanup | Ensures PG instances are stopped even on SLURM time-limit (SIGTERM) |
| venv bootstrap in every job | `uv sync` is a no-op when `uv.lock` is unchanged; simpler than a separate setup job |
| `slurm/run.sh` at repo root | Keeps scheduler scripts separate from `src/` and `scripts/` |

## Artifacts to create

- `slurm/run.sh` — SLURM job script
- `Dockerfile.pg-hpc-lite` and `Dockerfile.pg-hpc-full` — optional patched PostgreSQL images (only if UID permission issues arise)
