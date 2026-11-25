#!/usr/bin/env python3
"""Local Colab GPU Orchestrator for Split or Steal (Layer 2).

Wraps ``.venv/bin/colab`` (``google-colab-cli``) with strict compute-unit accounting
and guaranteed session teardown:
  1. Checks ``colab usage`` BEFORE allocating a GPU (records starting CU balance).
  2. Allocates an ephemeral GPU session (``colab new -s <session> --gpu <L4|A100|T4|H100>``).
  3. Captures LIVE ``colab usage`` burn rate (CUs/hr) while the GPU is active.
  4. Installs/verifies HF ML dependencies (``transformers``, ``peft``, ``trl``,
     ``bitsandbytes``, ``datasets``, ``accelerate``, ``safetensors``).
  5. Uploads training/verification scripts & character processed JSONL datasets.
  6. Streams remote execution logs live with ``--timeout 0``.
  7. Downloads JSON/MD reports into ``results/`` and LoRA adapter weights into
     ``checkpoints/`` (gitignored) via single compressed archives.
  8. GUARANTEES ``colab stop -s <session>`` inside a ``finally:`` block, checks
     final ``colab usage``, and emits an auditable ``cu_receipt_<session>.json``.

Never allocates a VM when ``--dry-run`` is passed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from training.utils import parse_colab_usage
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import parse_colab_usage  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COLAB_BIN = REPO_ROOT / ".venv" / "bin" / "colab"

REMOTE_WORK_ROOT = "/content/work"
COLAB_ML_PACKAGES: Tuple[str, ...] = (
    "transformers>=4.49.0",
    "peft>=0.14.0",
    "trl>=0.15.0",
    "bitsandbytes>=0.45.0",
    "datasets>=3.2.0",
    "accelerate>=1.3.0",
    "safetensors>=0.4.5",
)


def to_contents_api_path(remote_abs_path: str) -> str:
    """Convert ``/content/work/foo.py`` to ``content/work/foo.py`` for ``colab upload/download``."""
    return remote_abs_path.lstrip("/")


def build_remote_shell_runner(shell_cmd: str) -> str:
    """Build a small Python snippet for ``colab exec`` that streams a shell command cleanly."""
    return (
        "import subprocess, sys\n"
        f"cmd = {shell_cmd!r}\n"
        "print(f'[remote] $ {cmd}', flush=True)\n"
        "proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "assert proc.stdout is not None\n"
        "for line in proc.stdout:\n"
        "    sys.stdout.write(line)\n"
        "    sys.stdout.flush()\n"
        "rc = proc.wait()\n"
        "if rc != 0:\n"
        "    raise SystemExit(rc)\n"
    )


class ColabSessionRunner:
    """Managed wrapper around ``colab-cli`` with guaranteed teardown & CU receipts."""

    def __init__(
        self,
        colab_bin: str | Path = DEFAULT_COLAB_BIN,
        dry_run: bool = False,
    ):
        self.colab_bin = str(colab_bin)
        self.dry_run = dry_run
        self.planned_commands: List[List[str]] = []

    def _run_cmd(
        self,
        args: Sequence[str],
        *,
        stdin_text: Optional[str] = None,
        stream: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        cmd = [self.colab_bin, *args]
        self.planned_commands.append(list(cmd))
        if self.dry_run:
            print(f"[dry-run] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        if stream:
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                text=True,
                check=False,
            )
            if check and proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
            return proc

        proc = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n{msg}")
        return proc

    def get_usage(self) -> Dict[str, Any]:
        """Run ``colab usage`` and return parsed balance / rate / active assignments."""
        if self.dry_run:
            self.planned_commands.append([self.colab_bin, "usage"])
            return {
                "balance_cu": None,
                "usage_rate_cu_per_hr": None,
                "active_assignments": None,
                "parsed_ok": False,
                "raw": "<dry-run>",
            }
        res = self._run_cmd(["usage"], stream=False, check=True)
        return parse_colab_usage(res.stdout)

    def create_session(self, session: str, gpu: str, high_mem: bool = False) -> None:
        args = ["new", "-s", session, "--gpu", gpu]
        if high_mem:
            args.append("--high-mem")
        self._run_cmd(args, stream=True, check=True)

    def stop_session(self, session: str) -> None:
        self._run_cmd(["stop", "-s", session], stream=True, check=False)

    def exec_python(self, session: str, code: str, timeout: float = 0.0) -> None:
        self._run_cmd(
            ["exec", "-s", session, "--timeout", str(timeout)],
            stdin_text=code,
            stream=True,
            check=True,
        )

    def exec_shell(self, session: str, shell_cmd: str, timeout: float = 0.0) -> None:
        self.exec_python(session, build_remote_shell_runner(shell_cmd), timeout=timeout)

    def ensure_remote_dirs(self, session: str, dirs: Sequence[str]) -> None:
        lines = ["import os"] + [f"os.makedirs({d!r}, exist_ok=True)" for d in dirs]
        self.exec_python(session, "\n".join(lines) + "\n", timeout=60.0)

    def upload_file(self, session: str, local_path: str | Path, remote_abs_path: str) -> None:
        lp = Path(local_path)
        if not self.dry_run and not lp.is_file():
            raise FileNotFoundError(f"Local file not found for upload: {lp}")
        self._run_cmd(
            ["upload", "-s", session, str(lp), to_contents_api_path(remote_abs_path)],
            stream=True,
            check=True,
        )

    def download_file(self, session: str, remote_abs_path: str, local_path: str | Path) -> None:
        lp = Path(local_path)
        if not self.dry_run:
            lp.parent.mkdir(parents=True, exist_ok=True)
        self._run_cmd(
            ["download", "-s", session, to_contents_api_path(remote_abs_path), str(lp)],
            stream=True,
            check=True,
        )

    def download_dir_as_tarball(
        self,
        session: str,
        remote_dir: str,
        local_dest_dir: str | Path,
        tar_name: str,
    ) -> None:
        """Archive a remote directory into ``.tar.gz``, download it, and extract locally."""
        remote_tar = f"{REMOTE_WORK_ROOT}/{tar_name}.tar.gz"
        self.exec_shell(
            session,
            f"tar -czf {shlex.quote(remote_tar)} -C {shlex.quote(remote_dir)} .",
            timeout=300.0,
        )
        local_dest = Path(local_dest_dir)
        local_tar = local_dest.parent / f".{tar_name}.tar.gz"
        self.download_file(session, remote_tar, local_tar)
        if not self.dry_run:
            local_dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local_tar, "r:gz") as tf:
                tf.extractall(local_dest)
            local_tar.unlink(missing_ok=True)

    def install_ml_stack(self, session: str) -> None:
        pkgs = " ".join(shlex.quote(p) for p in COLAB_ML_PACKAGES)
        self.exec_shell(
            session,
            f"python3 -m pip install -q --upgrade {pkgs}",
            timeout=600.0,
        )


def _write_cu_receipt(
    receipt_path: Path,
    *,
    session: str,
    job_type: str,
    gpu: str,
    started_utc: str,
    elapsed_sec: float,
    usage_before: Dict[str, Any],
    usage_live: Optional[Dict[str, Any]],
    usage_after: Optional[Dict[str, Any]],
    status: str,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bal_before = usage_before.get("balance_cu")
    bal_after = usage_after.get("balance_cu") if usage_after else None
    cu_spent = (
        round(bal_before - bal_after, 4)
        if (bal_before is not None and bal_after is not None)
        else None
    )
    receipt: Dict[str, Any] = {
        "session": session,
        "job_type": job_type,
        "gpu": gpu,
        "status": status,
        "error": error,
        "started_utc": started_utc,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "usage_before": usage_before,
        "usage_live": usage_live,
        "usage_after": usage_after,
        "cu_spent_observed": cu_spent,
        "extra": extra or {},
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


# ---------------------------------------------------------------------------
# Job Commands
# ---------------------------------------------------------------------------


def cmd_usage(args: argparse.Namespace) -> int:
    runner = ColabSessionRunner(colab_bin=args.colab_bin, dry_run=False)
    info = runner.get_usage()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(
            f"Balance: {info['balance_cu']} CUs | "
            f"Burn Rate: {info['usage_rate_cu_per_hr']}/hr | "
            f"Active: {info['active_assignments']}"
        )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    runner = ColabSessionRunner(colab_bin=args.colab_bin, dry_run=args.dry_run)
    session = args.session or f"bench-{args.gpu.lower()}-{uuid.uuid4().hex[:5]}"
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.monotonic()

    usage_before = runner.get_usage()
    usage_live: Optional[Dict[str, Any]] = None
    usage_after: Optional[Dict[str, Any]] = None
    status = "ok"
    err_msg: Optional[str] = None

    print(
        f"[orchestrator:benchmark] session={session} gpu={args.gpu} "
        f"balance_before={usage_before.get('balance_cu')} CUs",
        flush=True,
    )

    try:
        runner.create_session(session, gpu=args.gpu, high_mem=args.high_mem)
        if not args.dry_run:
            time.sleep(3.0)
        usage_live = runner.get_usage()
        print(
            f"[orchestrator:benchmark] LIVE burn rate={usage_live.get('usage_rate_cu_per_hr')} CUs/hr",
            flush=True,
        )

        runner.ensure_remote_dirs(session, [REMOTE_WORK_ROOT])
        runner.upload_file(session, REPO_ROOT / "training" / "utils.py", f"{REMOTE_WORK_ROOT}/utils.py")
        runner.upload_file(
            session, REPO_ROOT / "training" / "bench_silicon.py", f"{REMOTE_WORK_ROOT}/bench_silicon.py"
        )
        if args.mode in ("auto", "hf"):
            runner.install_ml_stack(session)

        remote_out = f"{REMOTE_WORK_ROOT}/bench_{args.gpu.lower()}.json"
        bench_cmd = (
            f"python3 {REMOTE_WORK_ROOT}/bench_silicon.py "
            f"--mode {shlex.quote(args.mode)} "
            f"--model {shlex.quote(args.model)} "
            f"--quant {shlex.quote(args.quant)} "
            f"--steps {args.steps} --warmup {args.warmup} "
            f"--seq-len {args.seq_len} --batch-size {args.batch_size} "
            f"--gen-tokens {args.gen_tokens} "
            f"--output {shlex.quote(remote_out)}"
        )
        runner.exec_shell(session, bench_cmd, timeout=0.0)

        local_out = Path(args.output or (REPO_ROOT / "results" / f"bench_colab_{args.gpu.lower()}.json"))
        runner.download_file(session, remote_out, local_out)
    except BaseException as exc:
        status = "failed"
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        print(f"[orchestrator:benchmark] Guaranteed teardown: stopping '{session}'...", flush=True)
        runner.stop_session(session)
        if not args.dry_run:
            time.sleep(2.0)
        usage_after = runner.get_usage()
        elapsed = time.monotonic() - t0
        if not args.dry_run:
            receipt_path = REPO_ROOT / "results" / f"cu_receipt_{session}.json"
            receipt = _write_cu_receipt(
                receipt_path,
                session=session,
                job_type="benchmark",
                gpu=args.gpu,
                started_utc=started_utc,
                elapsed_sec=elapsed,
                usage_before=usage_before,
                usage_live=usage_live,
                usage_after=usage_after,
                status=status,
                error=err_msg,
                extra={"mode": args.mode, "model": args.model},
            )
            print(
                f"[orchestrator:benchmark] Receipt -> {receipt_path} | "
                f"CUs spent={receipt['cu_spent_observed']} | "
                f"Final balance={usage_after.get('balance_cu')} CUs",
                flush=True,
            )
    return 0


def cmd_train_character(args: argparse.Namespace) -> int:
    runner = ColabSessionRunner(colab_bin=args.colab_bin, dry_run=args.dry_run)
    character = args.character
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    valid_stages = {"cpt", "sft", "harvest_dpo", "dpo", "verify"}
    unknown = [s for s in stages if s not in valid_stages]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; allowed: {sorted(valid_stages)}")

    data_dir = Path(args.data_dir or (REPO_ROOT / "data" / "processed" / character))
    results_dir = Path(args.results_dir or (REPO_ROOT / "results" / character))
    ckpt_root = Path(args.checkpoints_dir or (REPO_ROOT / "checkpoints" / character))

    session = args.session or f"train-{character}-{uuid.uuid4().hex[:5]}"
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.monotonic()

    usage_before = runner.get_usage()
    usage_live: Optional[Dict[str, Any]] = None
    usage_after: Optional[Dict[str, Any]] = None
    status = "ok"
    err_msg: Optional[str] = None

    print(
        f"[orchestrator:train] character={character} stages={stages} gpu={args.gpu} "
        f"model={args.model} balance_before={usage_before.get('balance_cu')} CUs",
        flush=True,
    )

    remote_data = f"{REMOTE_WORK_ROOT}/data/{character}"
    remote_out = f"{REMOTE_WORK_ROOT}/out/{character}"

    try:
        runner.create_session(session, gpu=args.gpu, high_mem=args.high_mem)
        if not args.dry_run:
            time.sleep(3.0)
        usage_live = runner.get_usage()
        print(
            f"[orchestrator:train] LIVE burn rate={usage_live.get('usage_rate_cu_per_hr')} CUs/hr",
            flush=True,
        )

        runner.ensure_remote_dirs(session, [REMOTE_WORK_ROOT, remote_data, remote_out])
        runner.install_ml_stack(session)

        # Upload training & verification scripts
        for fname in ("utils.py", "train_character.py", "verify_weight_shift.py"):
            runner.upload_file(
                session, REPO_ROOT / "training" / fname, f"{REMOTE_WORK_ROOT}/{fname}"
            )

        # Upload available dataset JSONL files for this character
        candidate_files = (
            "cpt_train.jsonl",
            "sft_train.jsonl",
            "sft_val.jsonl",
            "val_raw.jsonl",
            "dpo_prompts.jsonl",
            "dpo_pairs.jsonl",
        )
        uploaded_files = set()
        for fname in candidate_files:
            local_f = data_dir / fname
            if args.dry_run or local_f.is_file():
                runner.upload_file(session, local_f, f"{remote_data}/{fname}")
                uploaded_files.add(fname)

        prev_adapter: Optional[str] = None
        produced_adapters: Dict[str, str] = {}

        if "cpt" in stages:
            cpt_out = f"{remote_out}/cpt"
            eval_flag = (
                f"--eval-file {shlex.quote(remote_data + '/val_raw.jsonl')}"
                if "val_raw.jsonl" in uploaded_files
                else ""
            )
            cmd = (
                f"python3 {REMOTE_WORK_ROOT}/train_character.py "
                f"--stage cpt --character {shlex.quote(character)} "
                f"--model {shlex.quote(args.model)} --quant {shlex.quote(args.quant)} "
                f"--train-file {shlex.quote(remote_data + '/cpt_train.jsonl')} {eval_flag} "
                f"--output-dir {shlex.quote(cpt_out)} "
                f"--max-steps {args.cpt_max_steps} --max-minutes {args.max_minutes_per_stage}"
            )
            runner.exec_shell(session, cmd, timeout=0.0)
            prev_adapter = cpt_out
            produced_adapters[f"{character}_cpt"] = cpt_out
            runner.download_file(
                session, f"{cpt_out}/training_metrics.json", results_dir / "cpt_training_metrics.json"
            )
            if not args.skip_download_weights:
                runner.download_dir_as_tarball(session, cpt_out, ckpt_root / "cpt", f"{character}_cpt")

        if "sft" in stages:
            sft_out = f"{remote_out}/sft"
            init_flag = f"--init-adapter {shlex.quote(prev_adapter)}" if prev_adapter else ""
            eval_flag = (
                f"--eval-file {shlex.quote(remote_data + '/sft_val.jsonl')}"
                if "sft_val.jsonl" in uploaded_files
                else ""
            )
            cmd = (
                f"python3 {REMOTE_WORK_ROOT}/train_character.py "
                f"--stage sft --character {shlex.quote(character)} "
                f"--model {shlex.quote(args.model)} --quant {shlex.quote(args.quant)} "
                f"--train-file {shlex.quote(remote_data + '/sft_train.jsonl')} {eval_flag} "
                f"{init_flag} --output-dir {shlex.quote(sft_out)} "
                f"--max-steps {args.sft_max_steps} --max-minutes {args.max_minutes_per_stage}"
            )
            runner.exec_shell(session, cmd, timeout=0.0)
            prev_adapter = sft_out
            produced_adapters[f"{character}_sft"] = sft_out
            runner.download_file(
                session, f"{sft_out}/training_metrics.json", results_dir / "sft_training_metrics.json"
            )
            if not args.skip_download_weights:
                runner.download_dir_as_tarball(session, sft_out, ckpt_root / "sft", f"{character}_sft")

        dpo_pairs_remote = f"{remote_data}/dpo_pairs.jsonl"
        if "harvest_dpo" in stages:
            dpo_pairs_remote = f"{remote_out}/dpo_pairs.jsonl"
            cmd = (
                f"python3 {REMOTE_WORK_ROOT}/train_character.py "
                f"--stage harvest_dpo --character {shlex.quote(character)} "
                f"--model {shlex.quote(args.model)} --quant {shlex.quote(args.quant)} "
                f"--train-file {shlex.quote(remote_data + '/dpo_prompts.jsonl')} "
                f"--output-dir {shlex.quote(remote_out)} "
                f"--output-file {shlex.quote(dpo_pairs_remote)} "
                f"--max-minutes {args.max_minutes_per_stage}"
            )
            runner.exec_shell(session, cmd, timeout=0.0)
            runner.download_file(session, dpo_pairs_remote, results_dir / "dpo_pairs_harvested.jsonl")
            runner.download_file(
                session,
                f"{remote_out}/harvest_dpo_metrics.json",
                results_dir / "harvest_dpo_metrics.json",
            )

        if "dpo" in stages:
            dpo_out = f"{remote_out}/dpo"
            init_flag = f"--init-adapter {shlex.quote(prev_adapter)}" if prev_adapter else ""
            cmd = (
                f"python3 {REMOTE_WORK_ROOT}/train_character.py "
                f"--stage dpo --character {shlex.quote(character)} "
                f"--model {shlex.quote(args.model)} --quant {shlex.quote(args.quant)} "
                f"--train-file {shlex.quote(dpo_pairs_remote)} "
                f"{init_flag} --output-dir {shlex.quote(dpo_out)} "
                f"--max-steps {args.dpo_max_steps} --dpo-beta {args.dpo_beta} "
                f"--max-minutes {args.max_minutes_per_stage}"
            )
            runner.exec_shell(session, cmd, timeout=0.0)
            prev_adapter = dpo_out
            produced_adapters[f"{character}_dpo"] = dpo_out
            runner.download_file(
                session, f"{dpo_out}/training_metrics.json", results_dir / "dpo_training_metrics.json"
            )
            if not args.skip_download_weights:
                runner.download_dir_as_tarball(session, dpo_out, ckpt_root / "dpo", f"{character}_dpo")

        if "verify" in stages and produced_adapters:
            verify_out = f"{remote_out}/verify"
            adapter_args = " ".join(
                shlex.quote(f"{k}={v}") for k, v in produced_adapters.items()
            )
            corpora_arg = (
                f"--corpora {shlex.quote(character + '=' + remote_data + '/val_raw.jsonl')}"
                if "val_raw.jsonl" in uploaded_files
                else ""
            )
            cmd = (
                f"python3 {REMOTE_WORK_ROOT}/verify_weight_shift.py "
                f"--model {shlex.quote(args.model)} --quant {shlex.quote(args.quant)} "
                f"--adapters {adapter_args} {corpora_arg} "
                f"--output-dir {shlex.quote(verify_out)}"
            )
            runner.exec_shell(session, cmd, timeout=0.0)
            runner.download_file(
                session, f"{verify_out}/verification_report.json", results_dir / "verification_report.json"
            )
            runner.download_file(
                session, f"{verify_out}/verification_summary.md", results_dir / "verification_summary.md"
            )

    except BaseException as exc:
        status = "failed"
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        print(f"[orchestrator:train] Guaranteed teardown: stopping '{session}'...", flush=True)
        runner.stop_session(session)
        if not args.dry_run:
            time.sleep(2.0)
        usage_after = runner.get_usage()
        elapsed = time.monotonic() - t0
        if not args.dry_run:
            receipt_path = results_dir / f"cu_receipt_{session}.json"
            receipt = _write_cu_receipt(
                receipt_path,
                session=session,
                job_type=f"train_character:{character}",
                gpu=args.gpu,
                started_utc=started_utc,
                elapsed_sec=elapsed,
                usage_before=usage_before,
                usage_live=usage_live,
                usage_after=usage_after,
                status=status,
                error=err_msg,
                extra={"character": character, "stages": stages, "model": args.model},
            )
            print(
                f"[orchestrator:train] Receipt -> {receipt_path} | "
                f"CUs spent={receipt['cu_spent_observed']} | "
                f"Final balance={usage_after.get('balance_cu')} CUs",
                flush=True,
            )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Upload multiple trained adapters & held-out corpora and run cross-character 4-stage verification."""
    runner = ColabSessionRunner(colab_bin=args.colab_bin, dry_run=args.dry_run)
    session = args.session or f"verify-{uuid.uuid4().hex[:5]}"
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.monotonic()

    out_dir = Path(args.output_dir or (REPO_ROOT / "results" / "verification"))
    usage_before = runner.get_usage()
    usage_live: Optional[Dict[str, Any]] = None
    usage_after: Optional[Dict[str, Any]] = None
    status = "ok"
    err_msg: Optional[str] = None

    try:
        runner.create_session(session, gpu=args.gpu, high_mem=args.high_mem)
        if not args.dry_run:
            time.sleep(3.0)
        usage_live = runner.get_usage()

        remote_adapters_root = f"{REMOTE_WORK_ROOT}/adapters"
        remote_corpora_root = f"{REMOTE_WORK_ROOT}/corpora"
        remote_verify_out = f"{REMOTE_WORK_ROOT}/verify_out"
        runner.ensure_remote_dirs(
            session, [REMOTE_WORK_ROOT, remote_adapters_root, remote_corpora_root, remote_verify_out]
        )
        runner.install_ml_stack(session)

        for fname in ("utils.py", "verify_weight_shift.py"):
            runner.upload_file(
                session, REPO_ROOT / "training" / fname, f"{REMOTE_WORK_ROOT}/{fname}"
            )

        remote_adapter_kv: List[str] = []
        for item in args.adapters:
            name, local_dir_str = item.split("=", 1)
            local_dir = Path(local_dir_str)
            r_dir = f"{remote_adapters_root}/{name}"
            runner.ensure_remote_dirs(session, [r_dir])
            for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
                lf = local_dir / fname
                if (args.dry_run and fname != "adapter_model.bin") or lf.is_file():
                    runner.upload_file(session, lf, f"{r_dir}/{fname}")
            remote_adapter_kv.append(f"{name}={r_dir}")

        remote_corpora_kv: List[str] = []
        for item in args.corpora:
            name, local_file_str = item.split("=", 1)
            r_file = f"{remote_corpora_root}/{name}_val_raw.jsonl"
            runner.upload_file(session, Path(local_file_str), r_file)
            remote_corpora_kv.append(f"{name}={r_file}")

        adapter_args = " ".join(shlex.quote(x) for x in remote_adapter_kv)
        corpora_args = (
            "--corpora " + " ".join(shlex.quote(x) for x in remote_corpora_kv)
            if remote_corpora_kv
            else ""
        )
        persona_args = (
            "--persona-prompts " + " ".join(shlex.quote(x) for x in args.persona_prompts)
            if args.persona_prompts
            else ""
        )

        cmd = (
            f"python3 {REMOTE_WORK_ROOT}/verify_weight_shift.py "
            f"--model {shlex.quote(args.model)} --quant {shlex.quote(args.quant)} "
            f"--adapters {adapter_args} {corpora_args} {persona_args} "
            f"--output-dir {shlex.quote(remote_verify_out)}"
        )
        runner.exec_shell(session, cmd, timeout=0.0)
        runner.download_file(
            session, f"{remote_verify_out}/verification_report.json", out_dir / "verification_report.json"
        )
        runner.download_file(
            session, f"{remote_verify_out}/verification_summary.md", out_dir / "verification_summary.md"
        )
    except BaseException as exc:
        status = "failed"
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        print(f"[orchestrator:verify] Guaranteed teardown: stopping '{session}'...", flush=True)
        runner.stop_session(session)
        if not args.dry_run:
            time.sleep(2.0)
        usage_after = runner.get_usage()
        elapsed = time.monotonic() - t0
        if not args.dry_run:
            receipt_path = out_dir / f"cu_receipt_{session}.json"
            _write_cu_receipt(
                receipt_path,
                session=session,
                job_type="verify_cross_corpus",
                gpu=args.gpu,
                started_utc=started_utc,
                elapsed_sec=elapsed,
                usage_before=usage_before,
                usage_live=usage_live,
                usage_after=usage_after,
                status=status,
                error=err_msg,
                extra={"adapters": args.adapters, "corpora": args.corpora},
            )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cost-guarded Colab GPU orchestrator with guaranteed teardown and CU receipts."
    )
    p.add_argument(
        "--colab-bin",
        default=str(DEFAULT_COLAB_BIN),
        help="Path to colab CLI binary (default: .venv/bin/colab).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # usage
    p_u = sub.add_parser("usage", help="Check current Colab CU balance & burn rate.")
    p_u.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_u.set_defaults(func=cmd_usage)

    # benchmark
    p_b = sub.add_parser("benchmark", help="Run Silicon Receipt micro-benchmark on Colab GPU.")
    p_b.add_argument("--gpu", default="L4", choices=["T4", "L4", "A100", "H100", "G4"])
    p_b.add_argument("--high-mem", action="store_true")
    p_b.add_argument("--session", default=None)
    p_b.add_argument("--mode", default="auto", choices=["auto", "synthetic", "hf"])
    p_b.add_argument("--model", default="Qwen/Qwen3-8B")
    p_b.add_argument("--quant", default="auto", choices=["auto", "none", "4bit"])
    p_b.add_argument("--steps", type=int, default=10)
    p_b.add_argument("--warmup", type=int, default=2)
    p_b.add_argument("--seq-len", type=int, default=512)
    p_b.add_argument("--batch-size", type=int, default=1)
    p_b.add_argument("--gen-tokens", type=int, default=128)
    p_b.add_argument("--output", default=None)
    p_b.add_argument("--dry-run", action="store_true")
    p_b.set_defaults(func=cmd_benchmark)

    # train-character
    p_t = sub.add_parser(
        "train-character",
        help="Run composable CPT -> SFT -> harvest_dpo -> DPO -> verify pipeline on Colab GPU.",
    )
    p_t.add_argument("--character", required=True, help="Character slug (e.g. machiavelli, gandhi).")
    p_t.add_argument(
        "--stages",
        default="cpt,sft,harvest_dpo,dpo,verify",
        help="Comma-separated ordered stages (default: cpt,sft,harvest_dpo,dpo,verify).",
    )
    p_t.add_argument("--gpu", default="A100", choices=["T4", "L4", "A100", "H100", "G4"])
    p_t.add_argument("--high-mem", action="store_true")
    p_t.add_argument("--session", default=None)
    p_t.add_argument("--model", default="Qwen/Qwen3-8B")
    p_t.add_argument("--quant", default="auto", choices=["auto", "none", "4bit"])
    p_t.add_argument("--data-dir", default=None)
    p_t.add_argument("--results-dir", default=None)
    p_t.add_argument("--checkpoints-dir", default=None)
    p_t.add_argument("--cpt-max-steps", type=int, default=250)
    p_t.add_argument("--sft-max-steps", type=int, default=200)
    p_t.add_argument("--dpo-max-steps", type=int, default=120)
    p_t.add_argument("--dpo-beta", type=float, default=0.1)
    p_t.add_argument("--max-minutes-per-stage", type=float, default=20.0)
    p_t.add_argument("--skip-download-weights", action="store_true")
    p_t.add_argument("--dry-run", action="store_true")
    p_t.set_defaults(func=cmd_train_character)

    # verify
    p_v = sub.add_parser(
        "verify",
        help="Run cross-character 4-Stage Weight-Verification Suite on Colab GPU.",
    )
    p_v.add_argument("--gpu", default="L4", choices=["T4", "L4", "A100", "H100", "G4"])
    p_v.add_argument("--high-mem", action="store_true")
    p_v.add_argument("--session", default=None)
    p_v.add_argument("--model", default="Qwen/Qwen3-8B")
    p_v.add_argument("--quant", default="auto", choices=["auto", "none", "4bit"])
    p_v.add_argument("--adapters", nargs="+", required=True, help="NAME=LOCAL_ADAPTER_DIR ...")
    p_v.add_argument("--corpora", nargs="*", default=[], help="NAME=LOCAL_VAL_RAW_JSONL ...")
    p_v.add_argument("--persona-prompts", nargs="*", default=[], help="NAME=PROMPT_TEXT ...")
    p_v.add_argument("--output-dir", default=None)
    p_v.add_argument("--dry-run", action="store_true")
    p_v.set_defaults(func=cmd_verify)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
