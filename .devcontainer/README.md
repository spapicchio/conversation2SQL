# Dev Container Setup

This directory configures the development environment using VS Code Dev Containers with Docker Compose. Three containers run together: the main development environment (VLLM + GPU), and two PostgreSQL instances with the BIRD-Interact databases.

> **How Docker Compose works here:** each service runs its own completely separate image — Docker Compose does not merge them. When VS Code attaches, you are inside the `verlai/verl` image only. The two PostgreSQL containers run alongside it, reachable over the network (`localhost:5432`, `localhost:5433`), but their filesystems are isolated. This is why tools like `psql` that live inside the database containers are not available in your shell by default. To address this, `postCreateCommand` installs `postgresql-client` into the dev container so `psql` and `pg_isready` are available directly without needing `docker exec`.

## Files

- `devcontainer.json` — VS Code Dev Containers configuration
- `docker-compose.yml` — multi-container orchestration
- `check_env.py` — environment sanity checks (run after rebuilding)

## Containers

### `devcontainer` — Main Development Environment

**Image:** `verlai/verl:vllm017.latest`

The primary container where all development happens. It is pre-installed with VLLM, Flash-Attention, and FlashInfer for LLM inference and RL training. VS Code attaches to this container.

Key settings:
- `network_mode: host` — shares the host network stack directly. This is required for VLLM and NCCL (multi-GPU communication). As a side effect, it can reach the PostgreSQL containers via `localhost` since their ports are published to the host.
- `runtime: nvidia` + `shm_size: 128g` — GPU access and large shared memory for multi-GPU workloads.
- `depends_on: [postgresql, postgresql_full]` — waits for both databases to start before the dev container becomes available.

Volumes mounted into this container:
| Host path | Container path | Purpose |
|---|---|---|
| `../` (repo root) | `/workspaces/conversation2SQL` | Source code |
| `/home/papicchi/.cache/huggingface` | `/hf_cache` | HuggingFace model cache |
| `/home/papicchi/data` | `/workspaces/conversation2SQL/data` | Training data |

After the container is created, `postCreateCommand` in `devcontainer.json` automatically installs:
- `gpustat` — GPU monitoring utility
- VeRL — editable install from `./verl` submodule
- `flashinfer-cubin` + `flashinfer-jit-cache` — optimized CUDA kernels for inference
- `postgresql-client` — PostgreSQL client tools (`psql`, `pg_isready`) for querying the databases directly from the dev container shell

### `postgresql` — BIRD-Interact Lite Database

**Image:** `docker.io/shawnxxh/bird-interact-postgresql:latest`
**Container name:** `bird_interact_postgresql`
**Host port:** `5432`

The lite version of the BIRD-Interact PostgreSQL database. Accessible from the dev container at:
```
postgresql://root:123123@localhost:5432/<database>
```

### `postgresql_full` — BIRD-Interact Full Database

**Image:** `docker.io/shawnxxh/bird-interact-postgresql-full:latest`
**Container name:** `bird_interact_postgresql_full`
**Host port:** `5433`

The full version of the BIRD-Interact PostgreSQL database. Accessible from the dev container at:
```
postgresql://root:123123@localhost:5433/<database>
```

Both PostgreSQL containers share the same configuration:
- `POSTGRES_USER`: `root`
- `POSTGRES_PASSWORD`: `123123`
- `max_connections`: 300
- `shared_buffers`: 256MB
- Data is persisted in named Docker volumes (`postgresql_data`, `postgresql_data_full`) so database contents survive container restarts.

## Networking

The dev container uses `network_mode: host`, meaning it is not on a Docker bridge network — it shares the host's network interface directly. The two PostgreSQL containers use standard bridge networking and publish their ports to the host (`5432` and `5433`). Because the dev container sees the host network, it can reach both databases at `localhost:5432` and `localhost:5433`.

```
┌─────────────────────────────────┐
│         Host network            │
│                                 │
│  devcontainer (network_mode:host)│
│                                 │
│    localhost:5432 ──────────────┼──► postgresql (bridge, port 5432)
│    localhost:5433 ──────────────┼──► postgresql_full (bridge, port 5433)
└─────────────────────────────────┘
```

## `docker-compose.yml` Field Reference

This section documents every field in `docker-compose.yml` so you can confidently edit it when something needs to change.

### `devcontainer` service

```yaml
image: verlai/verl:vllm017.latest
```
The Docker image for the main dev container. `verlai/verl` is the official VeRL image, pre-built with VLLM, Flash-Attention, FlashInfer, and CUDA 12.9. Change the tag (`vllm017.latest`) if you need a different version. Check available tags at [hub.docker.com/r/verlai/verl](https://hub.docker.com/r/verlai/verl).

---

```yaml
network_mode: host
```
Bypasses Docker's bridge networking — the container shares the host's network interface directly. **Do not change this.** VLLM requires it for multi-GPU NCCL communication, and it is the reason the dev container can reach the PostgreSQL containers at `localhost:5432` and `localhost:5433`.

---

```yaml
shm_size: '128g'
```
Shared memory (`/dev/shm`) available to the container. PyTorch multi-GPU training uses shared memory for inter-process communication. Reduce this only if the host has limited RAM and training fails to start, but values below `16g` are likely to cause OOM errors with large models.

---

```yaml
cap_add:
  - SYS_ADMIN
```
Grants the `SYS_ADMIN` Linux capability. Required by some VLLM and FlashInfer operations that need privileged access to kernel interfaces. Remove only if you have a specific security constraint and understand the consequences.

---

```yaml
runtime: nvidia
```
Selects the NVIDIA container runtime so that GPU devices are visible inside the container. Requires `nvidia-container-toolkit` to be installed on the host. Remove this line only if running on a CPU-only machine (VLLM inference will be unavailable).

---

```yaml
volumes:
  - ..:/workspaces/conversation2SQL:cached
  - /home/papicchi/.cache/huggingface:/hf_cache
  - /home/papicchi/data:/workspaces/conversation2SQL/data
```

These are **bind mounts** — they map a specific directory on the host directly into the container. The format is `<host_path>:<container_path>`.

| Line | Host path | Container path | Notes |
|---|---|---|---|
| 1 | `../` (repo root) | `/workspaces/conversation2SQL` | Source code. The `..` is relative to this `.devcontainer/` directory. Do not change the container path — VS Code expects it there. |
| 2 | `/home/papicchi/.cache/huggingface` | `/hf_cache` | HuggingFace model cache. **Change the host path** to match your user's cache directory, e.g. `/home/youruser/.cache/huggingface`. Without this, models will be re-downloaded on every container rebuild. |
| 3 | `/home/papicchi/data` | `/workspaces/conversation2SQL/data` | Training data directory. **Change the host path** to wherever your parquet training files live. The container path must stay as-is so the project config finds the data. |

The `:cached` flag on the first volume is a Docker performance hint (mainly relevant on macOS; ignored on Linux).

---

```yaml
depends_on:
  - postgresql
  - postgresql_full
```
Ensures both database containers are started before the dev container. This only checks that the containers have started, not that PostgreSQL is ready to accept connections. If you see connection errors on startup, wait a few seconds and retry — the DB may still be initialising.

---

```yaml
command: sleep infinity
```
Keeps the container alive indefinitely. VS Code attaches to the running container rather than running a one-shot command. Do not change this.

---

### `postgresql` service

```yaml
image: shawnxxh/bird-interact-postgresql:latest
```
Pre-loaded PostgreSQL image containing the BIRD-Interact lite database. Maintained by the BIRD-Interact team. If you need to rebuild the database from scratch, replace this with a plain `postgres:16` image and restore from a dump.

---

```yaml
container_name: bird_interact_postgresql
```
Explicit container name used for Docker CLI operations (e.g. `docker exec -it bird_interact_postgresql psql`). Must be unique on the host. Change it if there is a naming conflict with another container.

---

```yaml
environment:
  POSTGRES_USER: root
  POSTGRES_PASSWORD: 123123
  TZ: "Asia/Hong_Kong"
```

| Variable | Purpose | When to change |
|---|---|---|
| `POSTGRES_USER` | Superuser account name | If you need a different user. Update all connection strings accordingly. |
| `POSTGRES_PASSWORD` | Superuser password | Change to a stronger password in any non-local environment. Update connection strings to match. |
| `TZ` | Container timezone | Change to your local timezone (e.g. `Europe/Rome`) if timestamp columns matter for your use case. |

---

```yaml
volumes:
  - postgresql_data:/var/lib/postgresql/data
```

The format is `<volume_name>:<path_inside_container>`.

- `postgresql_data` — a **named volume** declared in the top-level `volumes:` block at the bottom of the file. Docker creates and manages this volume on the host (under `/var/lib/docker/volumes/`). Unlike a bind mount (which points to a specific host directory you choose), a named volume is fully managed by Docker — you don't pick the host path.
- `/var/lib/postgresql/data` — the directory PostgreSQL writes its data files to inside the container. This is fixed by the PostgreSQL image and should not be changed.

Because the data lives in a named volume (not inside the container's writable layer), it **survives `docker compose down`, image updates, and container rebuilds**. The database is only lost if the volume itself is deleted.

To wipe the database and start fresh:
```bash
docker volume ls                               # find the exact volume name (prefix is usually the compose project name)
docker volume rm devcontainer_postgresql_data  # delete it — next start will re-initialise from the image
```

---

```yaml
command:
  - "-c"
  - "max_connections=300"
  - "-c"
  - "shared_buffers=256MB"
```
PostgreSQL server flags passed at startup. Each `-c` introduces one `key=value` setting.

| Flag | Default in plain PG | Set to | When to change |
|---|---|---|---|
| `max_connections` | 100 | 300 | Increase if you hit "too many connections" errors when running parallel evaluations. Decrease to free RAM (each idle connection costs ~5–10 MB). |
| `shared_buffers` | 128MB | 256MB | Increase (e.g. `512MB`, `1GB`) if query performance is slow on the full database. Rule of thumb: 25% of available RAM. |

---

```yaml
ports:
  - "5432:5432"
```
The format is `<host_port>:<container_port>`. Maps host port `5432` to the container's PostgreSQL port (`5432` is the PostgreSQL default and is fixed by the image). Change the left-hand number (host port) if `5432` is already in use on your machine, e.g. `"15432:5432"`. Then update any connection strings that reference `localhost:5432`.

---

### `postgresql_full` service

Identical structure to `postgresql` with these differences:

| Field | `postgresql` | `postgresql_full` |
|---|---|---|
| `image` | `shawnxxh/bird-interact-postgresql:latest` | `shawnxxh/bird-interact-postgresql-full:latest` |
| `container_name` | `bird_interact_postgresql` | `bird_interact_postgresql_full` |
| `volumes` | `postgresql_data` | `postgresql_data_full` |
| `ports` | `5432:5432` | `5433:5432` |

The full image contains the complete BIRD-Interact dataset; the lite image is a smaller subset. Change the host-side port (`5433`) if it conflicts with another service.

---

### Top-level `volumes`

```yaml
volumes:
  postgresql_data:
  postgresql_data_full:
```
Declares the two named volumes used by the PostgreSQL services. Docker creates these automatically the first time `docker compose up` runs. The empty value (no options) means Docker uses its default local driver, storing data under `/var/lib/docker/volumes/` on the host. You can add a `driver_opts` block to store volumes on a specific disk or NFS mount if needed.

---

## Usage

### First time / Rebuild

Open the repo in VS Code and run:

```
Ctrl+Shift+P → Dev Containers: Rebuild and Reopen in Container
```

VS Code will pull all images, start all three containers via docker-compose, run `postCreateCommand`, and attach to the `devcontainer` service.

### Subsequent opens

```
Ctrl+Shift+P → Dev Containers: Reopen in Container
```

The named volumes retain all PostgreSQL data between sessions.

### Python environment (inside the dev container)

```bash
uv venv --system-site-packages   # create venv on top of system packages
uv sync                           # install project dependencies
source .venv/bin/activate
```

### Verifying the environment

After the container is up and the Python environment is activated, run the sanity-check script to confirm all dependencies, the project package, system tools, and both PostgreSQL connections are working:

```bash
# psycopg2 is needed for the database checks
python .devcontainer/check_env.py
```

All tests should show `OK`. If any fail, the error message will tell you what is missing and how to fix it.

### Checking the databases

`psql` and `pg_isready` are installed in the dev container by `postCreateCommand` (via `postgresql-client`), so you can run them directly in your shell.

**Check connectivity** (`pg_isready` exits 0 if the server is accepting connections):
```bash
pg_isready -h localhost -p 5432 -U root   # lite
pg_isready -h localhost -p 5433 -U root   # full
```

**List available databases** on each instance:
```bash
PGPASSWORD=123123 psql -h localhost -p 5432 -U root -c '\l'
PGPASSWORD=123123 psql -h localhost -p 5433 -U root -c '\l'
```

**List tables in a specific database** (replace `<database>` with a name from the output above):
```bash
PGPASSWORD=123123 psql -h localhost -p 5432 -U root -d <database> -c '\dt'
PGPASSWORD=123123 psql -h localhost -p 5433 -U root -d <database> -c '\dt'
```

### Updating the host paths for mounts

The volume mount paths in `docker-compose.yml` are set to `/home/papicchi/...`. If running on a different machine, update these two lines under the `devcontainer` service:

```yaml
- /home/papicchi/.cache/huggingface:/hf_cache
- /home/papicchi/data:/workspaces/conversation2SQL/data
```
