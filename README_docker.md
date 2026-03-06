# AndroidWorld Docker Guide

This guide covers full Docker usage for AndroidWorld: image build, container startup, sanity checks, and benchmark runs with `run_docker.py`.

---

## 1) Prerequisites

- Docker installed and running.
- Linux host is recommended.
- For better performance, enable hardware virtualization (KVM) on Linux.
- API keys only if you use model-based agents:
  - `OPENAI_API_KEY` for `m3a_gpt4v`
  - `GCP_API_KEY` for `m3a_gemini_gcp`

> Note: Docker support is experimental.

---

## 2) Build the Docker image

From repository root:

```bash
docker build -t android_world:latest .
```

Apple Silicon (build amd64 image):

```bash
docker buildx build --platform linux/amd64 -t android_world:latest .
```

---

## 3) Start the Docker environment server

Run the container (required privileges for emulator):

```bash
docker run --privileged --name android-world-server -p 5000:5000 -it android_world:latest
```

What happens in the container:
- Android emulator boots (headless)
- FastAPI server starts at `http://0.0.0.0:5000`

Keep this terminal running.

---

## 4) Verify server health

From another terminal on host:

```bash
curl http://localhost:5000/health
```

Expected response:

```json
{"status":"success"}
```

Optional smoke test client:

```bash
python scripts/run_suite_on_docker.py
```

---

## 5) Run benchmark with Docker backend

Use the new Docker runner:

```bash
python run_docker.py \
  --server_url=http://localhost:5000 \
  --suite_family=android_world \
  --agent_name=random_agent \
  --n_task_combinations=1 \
  --task_random_seed=30 \
  --output_path=./runs
```

### Common options

- `--server_url`: Docker server URL (default `http://localhost:5000`)
- `--suite_family`: one of
  - `android_world`
  - `miniwob`
  - `miniwob_subset`
  - `android`
  - `information_retrieval`
- `--tasks`: comma-separated subset of tasks
- `--n_task_combinations`: number of random parameter combinations per task
- `--task_random_seed`: task parameter seed
- `--max_steps`: max steps per task instance (default `10`)
- `--checkpoint_dir`: resume from existing run directory
- `--output_path`: parent directory for new run outputs

### Supported agents in Docker mode

- `random_agent`
- `human_agent`
- `m3a_gemini_gcp`
- `m3a_gpt4v`

Not supported in Docker mode:
- `t3a_gemini_gcp`
- `t3a_gpt4`
- `seeact`

Reason: current HTTP server does not expose full controller/UI-tree APIs required by those agents.

---

## 6) Example benchmark commands

### A) AndroidWorld full family with random agent

```bash
python run_docker.py \
  --suite_family=android_world \
  --agent_name=random_agent \
  --n_task_combinations=1
```

### B) Run only a task subset

```bash
python run_docker.py \
  --suite_family=android_world \
  --agent_name=random_agent \
  --tasks=ContactsAddContact,ClockStopWatchRunning
```

### C) MiniWoB family

```bash
python run_docker.py \
  --suite_family=miniwob \
  --agent_name=random_agent \
  --n_task_combinations=1
```

### D) Resume interrupted run

```bash
python run_docker.py \
  --suite_family=android_world \
  --agent_name=random_agent \
  --checkpoint_dir=./runs/run_20260305T120000000000
```

---

## 7) Output and checkpoints

Each task instance is saved incrementally as gzipped pickle:

- `<checkpoint_dir>/<TaskName>_<instance_id>.pkl.gz`

If `--checkpoint_dir` is omitted, a timestamped run directory is created under `--output_path`, for example:

- `./runs/run_20260305T121314123456/`

---

## 8) Running with model-based agents

For OpenAI:

```bash
export OPENAI_API_KEY=your_key
python run_docker.py --agent_name=m3a_gpt4v
```

For Gemini:

```bash
export GCP_API_KEY=your_key
python run_docker.py --agent_name=m3a_gemini_gcp
```

For local LLMs, set `OPENAI_BASE_URL` to your local server URL:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
python run_docker.py --agent_name=m3a_qwen3vl
```

---

## 9) Troubleshooting

### Server not healthy

- Check container logs in the server terminal.
- Emulator boot can take several minutes on first run.
- Ensure no port conflict on `5000`.

### Very slow performance

- Confirm virtualization/KVM support on host.
- Prefer Linux host for best performance.
- On Apple Silicon with amd64 emulation, performance is expected to be much slower.

### Permission or emulator startup issues

- Ensure `--privileged` is present in `docker run`.
- Retry with a clean container:

```bash
docker rm -f android-world-server
```

then start again.

---

## 10) Stop and cleanup

Stop running container:

```bash
docker stop android-world-server
```

Remove container:

```bash
docker rm android-world-server
```

Remove image (optional):

```bash
docker rmi android_world:latest
```
