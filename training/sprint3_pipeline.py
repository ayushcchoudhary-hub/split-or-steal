#!/usr/bin/env python3
"""Crash-Proof Sprint 3 Orchestrator (ADR-009): Per-Character Qwen3-14B Streaming + Round-Robin.

Enforces ADR-009 persistence so a Colab session timeout NEVER loses trained weights:
  1. Checks local `checkpoints/qwen14b/<char>/dpo_primary/adapter_model.safetensors` first;
     skips training any character already persisted locally and uploads it to the VM.
  2. Trains each remaining character (`CPT -> SFT -> Harvest -> 100% Fair Primary-Source DPO`)
     on the remote Colab GPU, then IMMEDIATELY streams its `dpo_primary/` adapter and
     `training_metrics.json` straight back to local disk before starting the next character.
  3. Runs the resumable 15-pair Multi-LoRA round-robin tournament (`sim/round_robin_14b.py`)
     and downloads `games.jsonl`, `simulations.db`, `tournament_metrics.json`, and
     `pairwise_matrix.json` locally.
  4. Stops the Colab session cleanly (`colab stop`) in a `finally` block if `--stop-on-exit`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_BIN = REPO_ROOT / ".venv" / "bin" / "colab"
SSH_KEY = Path("/tmp/colab_ed25519")

DEFAULT_CAST = [
    "machiavelli",
    "gandhi",
    "richard_nixon",
    "oscar_wilde",
    "warren_buffett",
    "van_gogh",
]


def _run(cmd: Sequence[str], *, check: bool = True, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, input=input_bytes)


def _colab_exec_py(session: str, code: str, timeout: int = 45) -> str:
    proc = subprocess.run(
        [str(COLAB_BIN), "exec", "-s", session, "--timeout", str(timeout)],
        input=code.encode("utf-8"),
        capture_output=True,
        check=True,
        timeout=timeout + 20,
    )
    return proc.stdout.decode("utf-8", errors="replace")


def _ssh_cmd(session: str, remote_cmd: str) -> List[str]:
    return [
        str(COLAB_BIN),
        "ssh",
        "-s",
        session,
        "--identity",
        str(SSH_KEY),
        "--",
        remote_cmd,
    ]


def _local_adapter_dir(char_id: str) -> Path:
    return REPO_ROOT / "checkpoints" / "qwen14b" / char_id / "dpo_primary"


def _has_local_adapter(char_id: str) -> bool:
    d = _local_adapter_dir(char_id)
    return (d / "adapter_model.safetensors").is_file() and (d / "adapter_config.json").is_file()


def _stream_dir_to_remote(session: str, local_paths: Sequence[str], remote_dest: str) -> None:
    """Stream repo-relative paths to remote_dest via tar over SSH stdin."""
    env = dict(os.environ, COPYFILE_DISABLE="1")
    tar_proc = subprocess.Popen(
        ["tar", "--no-xattrs", "-czf", "-", *local_paths],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        env=env,
    )
    ssh_proc = subprocess.Popen(
        _ssh_cmd(session, f"mkdir -p {remote_dest} && tar -xzf - -C {remote_dest}"),
        stdin=tar_proc.stdout,
    )
    assert tar_proc.stdout is not None
    tar_proc.stdout.close()
    rc_ssh = ssh_proc.wait()
    rc_tar = tar_proc.wait()
    if rc_ssh != 0 or rc_tar != 0:
        raise RuntimeError(f"Upload failed (tar={rc_tar}, ssh={rc_ssh}) for {local_paths}")


def _stream_remote_dir_to_local(session: str, remote_dir: str, local_dir: Path, excludes: Sequence[str] = ()) -> None:
    """Stream remote directory back to local_dir via tar over SSH stdout."""
    local_dir.mkdir(parents=True, exist_ok=True)
    ex_flags = " ".join(f"--exclude={e}" for e in excludes)
    ssh_proc = subprocess.Popen(
        _ssh_cmd(session, f"tar -czf - -C {remote_dir} {ex_flags} ."),
        stdout=subprocess.PIPE,
    )
    tar_proc = subprocess.Popen(
        ["tar", "-xzf", "-", "-C", str(local_dir)],
        stdin=ssh_proc.stdout,
    )
    assert ssh_proc.stdout is not None
    ssh_proc.stdout.close()
    rc_tar = tar_proc.wait()
    rc_ssh = ssh_proc.wait()
    if rc_ssh != 0 or rc_tar != 0:
        raise RuntimeError(f"Download failed (ssh={rc_ssh}, tar={rc_tar}) from {remote_dir}")


def _run_detached_and_wait(
    session: str,
    job_name: str,
    bash_script: str,
    poll_sec: int = 30,
    on_poll: object = None,
) -> None:
    """Run bash_script via detached nohup on Colab (immune to SSH drops) and poll via colab exec."""
    script_path = f"/content/job_{job_name}.sh"
    log_path = f"/content/job_{job_name}.log"
    done_path = f"/content/job_{job_name}.done"
    fail_path = f"/content/job_{job_name}.fail"

    wrapped = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"rm -f {done_path} {fail_path}\n"
        f"trap 'echo $? > {fail_path}' ERR\n"
        + bash_script
        + f"\ntouch {done_path}\n"
    )
    launch_py = f"""
import pathlib, subprocess
pathlib.Path({script_path!r}).write_text({wrapped!r}, encoding='utf-8')
pathlib.Path({script_path!r}).chmod(0o755)
p = subprocess.Popen(
    ['nohup', 'bash', {script_path!r}],
    stdout=open({log_path!r}, 'w'),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
print('PID:', p.pid)
"""
    out = _colab_exec_py(session, launch_py, timeout=45)
    print(f"[sprint3:{job_name}] Launched detached -> {out.strip()}", flush=True)

    t0 = time.monotonic()
    consec_poll_err = 0
    poll_idx = 0
    while True:
        time.sleep(poll_sec)
        poll_idx += 1
        check_py = f"""
import pathlib, subprocess
done = pathlib.Path({done_path!r}).exists()
fail = pathlib.Path({fail_path!r}).exists()
tail = ''
lp = pathlib.Path({log_path!r})
if lp.is_file():
    lines = lp.read_text(errors='replace').splitlines()
    tail = '\\n'.join(lines[-4:])
print('STATE:', 'DONE' if done else ('FAIL' if fail else 'RUNNING'))
print(tail)
"""
        try:
            status_out = _colab_exec_py(session, check_py, timeout=45)
            consec_poll_err = 0
        except Exception as exc:
            consec_poll_err += 1
            print(f"[sprint3:{job_name}] Transient poll error ({consec_poll_err}/4): {exc}", flush=True)
            if consec_poll_err >= 4:
                raise
            continue
        elapsed_m = (time.monotonic() - t0) / 60.0
        first_line = status_out.splitlines()[0] if status_out.splitlines() else ""
        last_line = status_out.splitlines()[-1] if status_out.splitlines() else ""
        print(f"[sprint3:{job_name} | {elapsed_m:.1f}m] {first_line} | {last_line[:110]}", flush=True)
        if callable(on_poll):
            try:
                on_poll(poll_idx)
            except Exception as exc:
                print(f"[sprint3:{job_name}] Heartbeat warning (non-fatal): {exc}", flush=True)
        if "STATE: DONE" in status_out:
            return
        if "STATE: FAIL" in status_out:
            full_tail = _colab_exec_py(
                session,
                f"import pathlib; print('\\n'.join(pathlib.Path({log_path!r}).read_text(errors='replace').splitlines()[-40:]))",
                timeout=30,
            )
            raise RuntimeError(f"Remote job {job_name} FAILED!\n{full_tail}")


def train_single_character(session: str, char_id: str, model: str) -> None:
    """Train one character CPT->SFT->Harvest->Fair DPO and immediately persist adapter locally."""
    out = f"/content/out/s3_14b/{char_id}"
    script = f"""
MODEL={model}
OUT={out}
mkdir -p $OUT

echo "=== [CPT 14B] {char_id} ==="
python3 /content/repo/training/train_character.py --stage cpt --character {char_id} --model $MODEL --quant 4bit \\
  --train-file /content/repo/data/processed/{char_id}/cpt_train.jsonl \\
  --output-dir $OUT/cpt \\
  --batch-size 1 --grad-accum 8 --max-seq-len 512 --lr 2e-5 --max-steps 25 --logging-steps 5 --max-minutes 10

echo "=== [SFT 14B] {char_id} ==="
python3 /content/repo/training/train_character.py --stage sft --character {char_id} --model $MODEL --quant 4bit \\
  --init-adapter $OUT/cpt \\
  --train-file /content/repo/data/processed/{char_id}/sft_train.jsonl \\
  --output-dir $OUT/sft \\
  --batch-size 1 --grad-accum 8 --max-seq-len 512 --lr 5e-5 --max-steps 30 --drop-system-prompt --logging-steps 5 --max-minutes 10

echo "=== [HARVEST FAIR DPO 14B] {char_id} ==="
python3 /content/repo/training/train_character.py --stage harvest_dpo --character {char_id} --model $MODEL --quant 4bit \\
  --train-file /content/repo/data/processed/{char_id}/dpo_primary_train.jsonl \\
  --output-dir $OUT \\
  --output-file $OUT/dpo_primary_train_pairs.jsonl \\
  --max-harvest 80 --harvest-batch-size 6 --harvest-max-new-tokens 160 --max-minutes 8
python3 /content/repo/training/train_character.py --stage harvest_dpo --character {char_id} --model $MODEL --quant 4bit \\
  --train-file /content/repo/data/processed/{char_id}/dpo_primary_val.jsonl \\
  --output-dir $OUT \\
  --output-file $OUT/dpo_primary_val_pairs.jsonl \\
  --harvest-batch-size 6 --harvest-max-new-tokens 160 --max-minutes 4

echo "=== [FAIR DPO 14B] {char_id} ==="
python3 /content/repo/training/train_character.py --stage dpo --character {char_id} --model $MODEL --quant 4bit \\
  --init-adapter $OUT/sft \\
  --train-file $OUT/dpo_primary_train_pairs.jsonl \\
  --eval-file $OUT/dpo_primary_val_pairs.jsonl \\
  --output-dir $OUT/dpo_primary \\
  --epochs 2.0 --max-steps 20 --batch-size 1 --grad-accum 8 --lr 3e-5 --dpo-beta 0.1 \\
  --max-seq-len 640 --drop-system-prompt --logging-steps 5 --max-minutes 12
"""
    _run_detached_and_wait(session, f"train_{char_id}", script, poll_sec=45)

    # ADR-009: IMMEDIATELY stream the trained adapter + metrics back to local disk!
    local_dir = _local_adapter_dir(char_id)
    print(f"[sprint3:{char_id}] Streaming dpo_primary adapter -> {local_dir} ...", flush=True)
    _stream_remote_dir_to_local(
        session,
        f"{out}/dpo_primary",
        local_dir,
        excludes=("trainer_state",),
    )
    # Also copy training_metrics.json into committed results/sprint3_adapters/
    res_dir = REPO_ROOT / "results" / "sprint3_adapters" / char_id
    res_dir.mkdir(parents=True, exist_ok=True)
    metrics_src = local_dir / "training_metrics.json"
    if metrics_src.is_file():
        (res_dir / "dpo_primary_metrics.json").write_text(metrics_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[sprint3:{char_id}] PERSISTED LOCALLY -> {local_dir}", flush=True)


def run_round_robin(session: str, cast: Sequence[str], model: str, games_per_pair: int, regimes: Sequence[str]) -> None:
    """Run resumable N-character Multi-LoRA round-robin and download results locally."""
    adapters_args = " ".join(f"{c}=/content/out/s3_14b/{c}/dpo_primary" for c in cast)
    chars_args = " ".join(cast)
    regimes_args = " ".join(regimes)
    out = "/content/out/s3_14b/tournament"

    # If local games.jsonl already exists from a previous session, upload it so runner resumes cleanly!
    local_tourn = REPO_ROOT / "results" / "sprint3_tournament_14b"
    if (local_tourn / "simulations.db").is_file() or (local_tourn / "runs").is_dir():
        print("[sprint3:tournament] Uploading existing local tournament state for seamless resume...", flush=True)
        _stream_dir_to_remote(session, ["results/sprint3_tournament_14b"], "/content/resume_tmp")
        _colab_exec_py(
            session,
            f"import subprocess; subprocess.run('mkdir -p {out} && cp -r /content/resume_tmp/results/sprint3_tournament_14b/* {out}/', shell=True, check=True)",
        )

    script = f"""
mkdir -p {out}
python3 /content/repo/sim/round_robin_14b.py \\
  --model {model} --quant 4bit \\
  --characters {chars_args} \\
  --characters-dir /content/repo/characters \\
  --adapters {adapters_args} \\
  --experiment sprint3_6cast_qwen14b_round_robin \\
  --games-per-pair {games_per_pair} --turns 2 \\
  --regimes {regimes_args} \\
  --output-dir {out}
"""
    def _snapshot(poll_idx: int) -> None:
        if poll_idx % 15 == 0:
            print(f"[sprint3:tournament] Periodic local snapshot (poll #{poll_idx}) -> {local_tourn}", flush=True)
            _stream_remote_dir_to_local(session, out, local_tourn)

    _run_detached_and_wait(session, "round_robin_14b", script, poll_sec=60, on_poll=_snapshot)
    print(f"[sprint3:tournament] Downloading full round-robin results -> {local_tourn} ...", flush=True)
    _stream_remote_dir_to_local(session, out, local_tourn)
    print(f"[sprint3:tournament] SAVED LOCALLY -> {local_tourn}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Crash-proof Sprint 3 Qwen3-14B orchestrator (ADR-009).")
    p.add_argument("--session", required=True, help="Active Colab session name (e.g. s3-14b).")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--characters", nargs="+", default=DEFAULT_CAST)
    p.add_argument("--games-per-pair", type=int, default=2)
    p.add_argument("--regimes", nargs="+", default=["prompt_only", "cpt_sft_dpo"])
    p.add_argument("--skip-tournament", action="store_true", help="Only train & persist adapters.")
    p.add_argument("--stop-on-exit", action="store_true", help="Always colab stop --session on exit.")
    args = p.parse_args(argv)

    try:
        # 1. Setup remote env + upload code & processed datasets
        print(f"[sprint3] Syncing repo & datasets to {args.session}...", flush=True)
        paths_to_sync = ["training", "sim", "characters"] + [
            f"data/processed/{c}" for c in args.characters
        ]
        _stream_dir_to_remote(args.session, paths_to_sync, "/content/repo")
        print(f"[sprint3] Installing Python dependencies (detached) on {args.session}...", flush=True)
        _run_detached_and_wait(
            args.session,
            "pip_deps",
            "pip install -q -U peft trl bitsandbytes datasets accelerate safetensors && python3 -c 'import peft, trl, bitsandbytes; print(\"DEPS_OK\", peft.__version__, trl.__version__)'",
            poll_sec=15,
        )

        # 2. Train missing characters & immediately stream each adapter locally (ADR-009)
        for cid in args.characters:
            if _has_local_adapter(cid):
                print(f"[sprint3:{cid}] Found cached local adapter -> uploading to {args.session}...", flush=True)
                _stream_dir_to_remote(
                    args.session,
                    [f"checkpoints/qwen14b/{cid}/dpo_primary"],
                    "/content/local_ckpt_sync",
                )
                _colab_exec_py(
                    args.session,
                    f"import subprocess; subprocess.run('mkdir -p /content/out/s3_14b/{cid} && cp -r /content/local_ckpt_sync/checkpoints/qwen14b/{cid}/dpo_primary /content/out/s3_14b/{cid}/', shell=True, check=True)",
                )
            else:
                train_single_character(args.session, cid, args.model)

        # 3. Run resumable round-robin tournament
        if not args.skip_tournament:
            run_round_robin(args.session, args.characters, args.model, args.games_per_pair, args.regimes)
    finally:
        if args.stop_on_exit:
            print(f"[sprint3] Guaranteed teardown: stopping {args.session}...", flush=True)
            subprocess.run([str(COLAB_BIN), "stop", "-s", args.session], check=False)
            subprocess.run([str(COLAB_BIN), "usage"], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
