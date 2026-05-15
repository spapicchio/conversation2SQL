# HPC Singularity Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `slurm/run.sh` to run the conversation2SQL evaluation pipeline on an HPC cluster using Singularity, plus optional patched PostgreSQL Dockerfiles for UID permission fallback.

**Architecture:** Three Singularity instances run inside a single SLURM job — `pg_lite` on port 5432, `pg_full` on port 5433 (explicit `-c port=5433` to avoid host-network collision), and the `verl` compute container with GPU passthrough. Cleanup is guaranteed via `trap EXIT`. SIF files are already built at `sif/`.

**Tech Stack:** Bash, SLURM (`#SBATCH`), Singularity (`instance start/stop`, `exec --nv`), `uv`, Docker (fallback Dockerfiles only)

---

## File Map

| Action | Path | Purpose |
|---|---|---|
| Create | `slurm/run.sh` | SLURM job script — starts PG instances, bootstraps venv, runs experiment |
| Create | `Dockerfile.pg-hpc-lite` | Patched pg_lite image with relaxed PGDATA permissions (fallback only) |
| Create | `Dockerfile.pg-hpc-full` | Patched pg_full image with relaxed PGDATA permissions (fallback only) |

---

## Task 1: Create `slurm/run.sh`

**Files:**
- Create: `slurm/run.sh`

- [ ] **Step 1: Create the `slurm/` directory and write `run.sh`**

```bash
mkdir -p /workspaces/conversation2SQL/slurm
```

Write `slurm/run.sh` with the following content:

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

# --- 0. Bootstrap venv (fast no-op if uv.lock unchanged) ---
singularity exec \
  --bind $REPO:/workspaces/conversation2SQL \
  $SIF/verl.sif bash -c "
    cd /workspaces/conversation2SQL
    uv venv --system-site-packages
    uv sync
  "

# --- 1. Start PostgreSQL lite (port 5432) ---
singularity instance start --writable-tmpfs \
  $SIF/pg_lite.sif pg_lite_${ID} \
  -c max_connections=300 -c shared_buffers=256MB

# --- 2. Start PostgreSQL full (port 5433) ---
# Must pass -c port=5433 explicitly: Singularity uses host network so both
# instances would otherwise collide on port 5432.
singularity instance start --writable-tmpfs \
  $SIF/pg_full.sif pg_full_${ID} \
  -c port=5433 -c max_connections=300 -c shared_buffers=256MB

# --- 3. Wait for both to accept connections ---
echo "Waiting for PostgreSQL lite (5432)..."
until nc -z localhost 5432 2>/dev/null; do sleep 2; done
echo "Waiting for PostgreSQL full (5433)..."
until nc -z localhost 5433 2>/dev/null; do sleep 2; done
echo "PostgreSQL ready."

# --- 4. Run experiment ---
# Extra flags passed to sbatch are forwarded: sbatch run.sh --baseline no_tool
singularity exec --nv \
  --bind $REPO:/workspaces/conversation2SQL \
  --bind $HF:/hf_cache \
  --env HF_HOME=/hf_cache \
  $SIF/verl.sif \
  bash -c 'cd /workspaces/conversation2SQL && source .venv/bin/activate && python main.py --config configs/eval_pipeline_config.yaml "$@"' \
  -- "$@"
```

- [ ] **Step 2: Make it executable and verify syntax**

```bash
chmod +x /workspaces/conversation2SQL/slurm/run.sh
bash -n /workspaces/conversation2SQL/slurm/run.sh
```

Expected: no output (clean syntax).

- [ ] **Step 3: Commit**

```bash
git add slurm/run.sh
git commit -m "feat: add slurm/run.sh for HPC Singularity deployment"
```

---

## Task 2: Create fallback Dockerfiles for PostgreSQL UID permission fix

These are only needed if the PostgreSQL Singularity instances fail to start on the HPC due to the data directory being owned by the `postgres` OS user (UID 70 or 999), which your HPC user cannot write to. Skip building these until you hit that error.

**Files:**
- Create: `Dockerfile.pg-hpc-lite`
- Create: `Dockerfile.pg-hpc-full`

- [ ] **Step 1: Write `Dockerfile.pg-hpc-lite`**

```dockerfile
FROM shawnxxh/bird-interact-postgresql:latest
RUN chmod -R 777 /var/lib/postgresql/data
```

Save to `/workspaces/conversation2SQL/Dockerfile.pg-hpc-lite`.

- [ ] **Step 2: Write `Dockerfile.pg-hpc-full`**

```dockerfile
FROM shawnxxh/bird-interact-postgresql-full:latest
RUN chmod -R 777 /var/lib/postgresql/data
```

Save to `/workspaces/conversation2SQL/Dockerfile.pg-hpc-full`.

- [ ] **Step 3: Verify Dockerfile syntax**

```bash
docker build --no-cache --dry-run -f /workspaces/conversation2SQL/Dockerfile.pg-hpc-lite . 2>&1 | head -5
docker build --no-cache --dry-run -f /workspaces/conversation2SQL/Dockerfile.pg-hpc-full . 2>&1 | head -5
```

Expected: `#1 [internal] load build definition` (BuildKit dry-run output, no errors).

If `--dry-run` is not supported by your Docker version, skip this step and verify at actual build time on your local machine.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.pg-hpc-lite Dockerfile.pg-hpc-full
git commit -m "feat: add patched PostgreSQL Dockerfiles for HPC UID permission fallback"
```

---

## HPC deployment steps (not git tasks — run on HPC)

These steps are run manually on the HPC after the above commits are on `$SCRATCH`.

### One-off venv bootstrap (interactive SLURM job)

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
exit
```

### Submit an experiment

```bash
# Default config
sbatch $SCRATCH/conversation2SQL/slurm/run.sh

# Override baseline and enable debug mode
sbatch $SCRATCH/conversation2SQL/slurm/run.sh --baseline no_tool --debug true

# Full BIRD-Interact agent
sbatch $SCRATCH/conversation2SQL/slurm/run.sh --baseline bird_full
```

### If PostgreSQL fails due to UID permissions

On your **local machine**:

```bash
cd /workspaces/conversation2SQL

docker build -f Dockerfile.pg-hpc-lite -t pg-lite-hpc .
docker build -f Dockerfile.pg-hpc-full -t pg-full-hpc .

singularity build pg_lite_patched.sif  docker-daemon://pg-lite-hpc
singularity build pg_full_patched.sif  docker-daemon://pg-full-hpc

rsync -avP pg_lite_patched.sif pg_full_patched.sif <user>@<hpc-login>:$SCRATCH/sif/
```

Then edit `slurm/run.sh` to point `pg_lite.sif` → `pg_lite_patched.sif` and `pg_full.sif` → `pg_full_patched.sif`.
