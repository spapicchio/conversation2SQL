# conversation2SQL

# Installation

We are using the base image provided by [VeRL](https://verl.readthedocs.io/en/latest/start/install.html#install-from-docker-image): https://hub.docker.com/r/vllm/vllm-openai.

It already contains VLLM and Flash-Attn:

```bash
python3 -c "import verl; print(verl.__version__); print(verl.__file__)"
>> 0.8.0.dev
>> /workspaces/conversation2SQL/verl/verl/__init__.py

python3 -c "import flash_attn; print(flash_attn.__version__); print(flash_attn.__file__)"
>> 2.8.1
>> /usr/local/lib/python3.12/dist-packages/flash_attn/__init__.py
```

Please Note that all the system packages are managed in the devcontainer.json files.

Note that verl is installed as a git submodule/clone, 
if you want to update verl you can ```git pull``` inside verl.

```bash
flashinfer show-config
# flashinfer configurations with JIT and CUBIN installed 
>>  === Version Info ===
>> FlashInfer version: 0.6.3
>> flashinfer-cubin version: 0.6.3
>> flashinfer-jit-cache version: 0.6.3+cu129
>> === Torch Version Info ===
>> Torch version: 2.9.1+cu129
>> CUDA runtime available: Yes
```

Okay now we can create an UV venv based on the system-site-packages:

```bash
# Create the venv based on the system site packages directory
uv venv --system-site-packages
# install custom packages
uv sync 
# activate env
source .venv/bin/activate
```
