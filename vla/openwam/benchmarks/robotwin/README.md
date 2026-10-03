# RoboTwin Benchmark Evaluation

These scripts assume the OpenWAM policy server is **already running**. They only cover the RoboTwin side of the evaluation loop.

## Files

| File | Description |
|---|---|
| `openwam2robotwin_interface.py` | RoboTwin client — talks to the WebSocket server. |
| `policy_config.yml` | Config template; `host` / `port` are injected at runtime. |
| `single_eval.sh` | Run evaluation on a single task (whole-task, RoboTwin's native loop). |
| `multi_eval.sh` | Run evaluation on multiple tasks sequentially. |
| `dispatcher.py` | Central **episode-level** scheduler (TCP): per-`(task,mode)` seed allocator + dynamic env-affinity/duplication + result aggregation. Has a `--self-test` fleet simulation and an optional live status HTTP endpoint. |
| `episode_worker.py` | One worker process = one `(task,mode)` assignment: boots the env once, then streams that task's episodes from the dispatcher. `--dry-run` simulates episodes with no RoboTwin. |
| `episode_eval.sh` | Env-setup shim (EGL/PYTHONPATH/CUDA) that execs one `episode_worker.py`. |
| `parallel_eval.sh` | Single-machine multi-GPU eval against already-running servers, using episode-level dynamic scheduling (dispatcher + one supervisor loop per GPU). |
| `export_results_csv.py` | Export to CSV. Prefers the dispatcher's `results.jsonl` (per-episode); falls back to `summary.tsv` + per-task `Success rate` grepping for legacy runs. |
| `step_limits.yml` | Per-task `step_lim` overrides (see below). |

## Episode-level dynamic scheduling

`parallel_eval.sh` schedules a **single episode** as
the unit of work, not a whole task. This eliminates the tail-idle waste of the
old whole-task queue: when there are more free GPUs than unstarted tasks, idle
GPUs join an in-progress task (spawn a duplicate env) to drain its remaining
episodes in parallel, so no GPU sits idle while any episode remains.

How it works:

- A central **dispatcher** (`dispatcher.py`; TCP) owns, per
  `(task, mode)`, a monotonic **seed allocator** and the episode counters. Every
  raw seed is handed out at most once globally, so no scene is ever evaluated
  twice (RoboTwin scenes are fully determined by their integer seed).
- Each GPU **slot** runs a supervisor loop that keeps launching
  `episode_worker.py`. A worker claims one `(task,mode)`, boots its RoboTwin env
  **once**, and streams that task's episodes: `request_seed` → expert-check →
  (valid) `request_commit` → policy rollout → `report_result`. The commit
  handshake makes each job land on **exactly `--test-num` episodes** (no
  overshoot). When the dispatcher drains the job, the worker exits and the
  supervisor launches a fresh one for the next assignment.
- **Duplication policy**: an idle slot first starts any unstarted task; when
  none remain it joins the in-progress task with the longest ETA, but **only if**
  that task still has at least `--min-remaining-for-dup` (θ) episodes left and is
  under its env cap (`ceil(remaining/θ)`). This avoids booting an env that the
  existing env(s) would finish before the new one is even ready.

New flags (both scripts):

| Flag | Default | Description |
|---|---:|---|
| `--test-num` | `100` | Episodes per `(task,mode)`. |
| `--seed` | `0` | Base seed; `st_seed = 100000*(1+seed)`, matching RoboTwin. |
| `--min-remaining-for-dup` | `8` | θ: don't spawn a new env for a task with fewer remaining episodes. |
| `--no-dup` | off | Strict mode: exactly one env per task, never duplicate (most reproducible; equivalent to the old whole-task granularity per job). |
| `--dispatch-port` | `8790` | Dispatcher TCP port. |
| `--http-port` | `0` | Serve a live status page (`/` HTML, `/api/state` JSON); `0` = off. |

Liveness / watchdog (the dispatcher never hangs silently):

- **Stall abort** — if no worker sends any request for `--stall-timeout` seconds
  (default 1800), or zero workers are connected for `--idle-grace` seconds
  (default 120) while jobs remain, the dispatcher writes an `incomplete`
  `summary.tsv` and exits non-zero (2) instead of self-spinning forever (covers
  every slot retiring, or all workers going silent).
- **Hung-worker reclaim** — a worker emits a keep-alive heartbeat every ~50
  rollout steps, so a genuinely hung one (socket open but silent for
  `--worker-timeout` seconds, default 1200) has its in-flight seed returned and
  env slot freed for others. Thanks to the heartbeat this need only exceed ~50
  inference steps, not a whole episode. A late report from a revived worker is
  ignored (no double count).
- **Give-up cap** — a task whose expert-check almost never passes stops after
  `target × --max-attempt-factor` seed attempts (default 50; `0` = unlimited),
  is marked `exhausted` in `summary.tsv`, and the run still terminates (exit 3).
- On exit (complete / exhausted / stall) the dispatcher touches a shared
  `.done` file; each node's launcher reaps its local (even wedged) workers on
  that signal, so no node's `wait` hangs on a stuck sim process.

Determinism note: the first `test_num` valid episodes of each task use the same
scenes as an upstream single-process run (seeds are handed out in order and a
seed's validity is policy-independent); only the assignment of episodes to GPUs
is non-deterministic. Use `--no-dup` for the most reproducible, single-stream
behavior.

Outputs land in the log directory: `results.jsonl` (one line per completed
episode — the authoritative record), `summary.tsv` (per-`(task,mode)` success
rate), `state.json` (live snapshot, rewritten atomically as the run progresses),
and `run.env` (parameters). Turn them into a CSV with `export_results_csv.py`.

Live monitoring:

- **`web_control.py`** (recommended) reads `state.json` /
  `results.jsonl` straight from the shared log directory — no network path to the
  compute nodes needed:
  `python benchmarks/web_control.py <log_dir> --benchmark robotwin --port 8765`.
- **`--http-port`** on the dispatcher serves the same live data directly, but is
  **off by default**; `web_control.py` on the shared log directory is the simpler choice.

### Dry-run (no GPUs / no RoboTwin)

Both the scheduling logic and the whole orchestration can be exercised without a
simulator:

```bash
# Fleet simulation: models env-boot cost + episode time, prints dup-vs-no-dup
# makespan/utilization and asserts exact counts + zero duplicate seeds.
python benchmarks/robotwin/dispatcher.py --self-test

```

## Per-task step_lim overrides

RoboTwin ships upstream per-task step limits in `task_config/_eval_step_limit.yml`. To tweak them without patching the RoboTwin source tree, edit [`step_limits.yml`](step_limits.yml) in this directory:

```yaml
# step_limits.yml (values here match what is checked in)
adjust_bottle: 160
open_laptop: 288
put_bottles_dustbin: 640
```

Semantics:

- Any task listed here overrides RoboTwin's upstream value for that task.
- Tasks not listed keep RoboTwin's original value (which itself falls back to `1000` when the upstream file also lacks the task).
- The file is loaded once at adapter import and applied at the first step of each episode via `TASK_ENV.step_lim = <override>`. Edits to the YAML only take effect in a fresh eval process — restart the evaluation after tweaking values.
- Leaving the file empty (comments only) reproduces stock RoboTwin behavior.

## Environment Setup

### 1. Install the RoboTwin environment

Follow the [official RoboTwin installation guide](https://github.com/RoboTwin-Platform/RoboTwin) to clone the repo, create the Conda environment, install dependencies, and download assets. When you're done you should have a working RoboTwin Conda environment (default name `robotwin`) and a local checkout of the RoboTwin repository.

### 2. Set environment variables

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin      # RoboTwin repo root (required)

export ROBOTWIN_ENV=robotwin                # RoboTwin Conda env name (default: robotwin)
```

### 3. Match the checkpoint action/state mode

`policy_config.yml` must match the checkpoint's saved `config.yaml`:

- `dataloader.action_mode: eef` / `architecture.state_dim: 20` → keep `action_type: ee`, `state_dim: 20`.
- `dataloader.action_mode: joint` / `architecture.state_dim: 14` → set `action_type: qpos`, `state_dim: 14`.
- Keep `send_state: true` for any checkpoint with `architecture.use_proprioception: true`; the adapter will fail fast if the extracted RoboTwin state dimension is wrong.

### 4. Start the OpenWAM server

Start the server separately before running any evaluation (it can live on a remote machine — just make sure the host/IP and the port are reachable):

```bash
bash scripts/deploy.sh --ckpt-dir /path/to/ckpt_dir --port XXXX
```

## Usage

### Single-task evaluation

**Invocation:**

```bash
bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [port] [host]
```

| Argument | Description |
|---|---|
| `task_name` | RoboTwin task name (e.g. `adjust_bottle`). |
| `task_config` | `demo_clean` or `demo_randomized`. |
| `ckpt_setting` | Label written into result filenames (e.g. `openwam`). |
| `gpu_id` | CUDA device for the RoboTwin simulator. |
| `port` | OpenWAM server port (default: `8848`). |
| `host` | OpenWAM server address (default: `127.0.0.1`). |

**Example:**

```bash
bash single_eval.sh adjust_bottle demo_clean openwam 0 8848 127.0.0.1
```


### Multi-task evaluation

```bash
bash multi_eval.sh -m <mode> -n <name> -d <ckpt_dir> [options] <tasks...>
```

**Required flags:**

| Flag | Description |
|---|---|
| `-m`, `--mode` | `demo_clean` or `demo_randomized`. |
| `-n`, `--name` | Label used for the log directory. |
| `-d`, `--ckpt-dir` | OpenWAM checkpoint directory used for log placement and run labeling; the evaluator still talks to an already-running server and does not load weights. |

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | OpenWAM server address. |
| `--port` | `8848` | OpenWAM server port. |
| `-g`, `--gpu` | `0` | CUDA device for the RoboTwin simulator. |

**Examples:**

```bash
# Run two named tasks
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpt_dir \
    adjust_bottle open_laptop

# Run all 50 RoboTwin 2.0 tasks
bash multi_eval.sh -m demo_randomized -n run1 -d /path/to/ckpt_dir all

# Point at a remote server
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpt_dir \
    --host 192.168.1.10 --port 8768 all

# Read the task list from a file (one task per line, `#` comments supported)
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpt_dir tasks.txt
```

### Export evaluation results to CSV

Use `export_results_csv.py` after a run to produce a single CSV. For
episode-level runs it aggregates the authoritative per-episode `results.jsonl`
(success / episodes / step-limit hits per `(task,mode)`); if that file is absent
(legacy whole-task runs) it falls back to `summary.tsv` plus per-task
`Success rate` log grepping.

**Invocation:**

```bash
python benchmarks/robotwin/export_results_csv.py \
    /path/to/log_dir \
    -o /path/to/log_dir/results.csv
```

If `-o` is omitted, the default output is:

```text
<log_dir>/results.csv
```

The CSV columns are:

| Column | Description |
|---|---|
| `policy_name` | Value passed with `-n`, `--name`. |
| `requested_mode` | Original `-m`, `--mode` value. |
| `task` | RoboTwin task name. |
| `mode` | Concrete task config: `demo_clean` or `demo_randomized`. |
| `node` | Node rank (blank for episode-level runs — a task's episodes span nodes/workers). |
| `worker` | Local worker index (blank for episode-level runs, same reason). |
| `status` | `ok` (target reached), `exhausted` (gave up per max-attempts), or `incomplete`. Legacy runs: `ok`/`failed`. |
| `exit_code` | Task process exit code (blank for episode-level runs). |
| `success_rate` | Success rate. Episode-level: `successes/episodes` from `results.jsonl` (percent). Legacy: parsed from the task log. |
| `episodes` | Completed episodes for the `(task,mode)`. Episode-level: count in `results.jsonl`. Legacy: `Success!`/`Fail!` verdicts in the log. |
| `step_limit_hits` | Episodes truncated at `step_lim` (ran out of steps rather than reaching a terminal state) — **not necessarily model errors**; a high count means `step_lim` may be too tight (see `step_limits.yml`). |
| `log_path` | `results.jsonl` for episode-level runs; the per-task log for legacy runs. |

The exporter also validates completeness when `run.env` is available: episode-level runs flag any `(task,mode)` that reached fewer than `test_num` episodes; legacy runs report duplicate/missing/unexpected rows.

Strict parsing mode returns a non-zero exit code if any task log is missing, does not contain a parseable success rate, or fails the completeness checks above:

```bash
python benchmarks/robotwin/export_results_csv.py /path/to/log_dir --strict
```

## FAQ

### Render Error (headless servers)

On a headless Linux box the SAPIEN renderer fails to find an X display and raises `Render Error`. Start a virtual framebuffer (Xvfb):

```bash
sudo apt-get install -y xvfb   # if not installed yet
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
bash single_eval.sh adjust_bottle demo_clean openwam 0 8848 127.0.0.1
```

Or, in one step, use `xvfb-run`:

```bash
xvfb-run -a bash single_eval.sh adjust_bottle demo_clean openwam 0 8848 127.0.0.1
```

---

### `policy_config.yml` options

```yaml
# Observation settings
# The client always forwards RoboTwin's head / left / right cameras to the server.
# The server inspects the checkpoint's saved config.yaml to decide:
#   - multiview=false → single-view preprocessing using head_camera only
#   - multiview=true  → composed into the L-shape multi-view layout used at training time
# The client no longer needs to configure camera selection or resolution.
send_state: true          # Include the proprio state vector in the obs message.
state_dim: 20             # Fail-fast expected dim. 20 for eef/ee, 14 for joint/qpos.
request_timeout: 300      # WebSocket timeout in seconds.

# Action settings (must match the `action_mode` used at training time).
action_type: ee           # ee   — EEF mode (action_mode: eef at train time, default).
                          #        Server returns 20D (xyz + rot6d + grip) × 2,
                          #        auto-converted to 16D (xyz + quat + grip) × 2 before dispatch.
                          # qpos — Joint-angle mode (action_mode: joint at train time).
                          #        Server returns 14D, passed straight to take_action.

# action_indices: null    # Optional index reordering for the returned action vector (null = no reorder).
```
