#!/usr/bin/env python3
"""Run CPU TSAN tests, retrying only recognized pre-main address-layout failures."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Clang 18 reports the denied personality syscall explicitly instead of GCC's
# occasional empty startup SIGSEGV. Never retry a bare signal, test error or race.
CLANG_LAYOUT_FAILURE = re.compile(
    r'ThreadSanitizer: CHECK failed: tsan_platform_linux\.cpp:\d+ '
    r'"\(\(personality\(old_personality \| ADDR_NO_RANDOMIZE\)\)\) != \(\(-1\)\)" '
    r'\(0xffffffffffffffff, 0xffffffffffffffff\) \(tid=\d+\)\n?'
)
GCC_LAYOUT_FAILURE = re.compile(
    r"FATAL: ThreadSanitizer: unexpected memory mapping 0x[0-9a-f]+-0x[0-9a-f]+\n?"
)


def is_layout_failure(result):
    if result.stdout:
        return False
    return (
        result.returncode == -11 and CLANG_LAYOUT_FAILURE.fullmatch(result.stderr) is not None
    ) or (
        result.returncode == 66 and GCC_LAYOUT_FAILURE.fullmatch(result.stderr) is not None
    )


def run(command, expected_warning=None, suppress_gdal=True, max_attempts=64):
    environment = os.environ.copy()
    # This CPU-only child does not use PyTorch/HPC-X. Leave the container's preload
    # unchanged for all other commands, including Python export and GPU inference.
    environment.pop("LD_PRELOAD", None)
    options = ["halt_on_error=1", "exitcode=66", "second_deadlock_stack=1",
               "allow_addr2line=1", "external_symbolizer_path=/usr/bin/addr2line"]
    if suppress_gdal:
        options.append(f"suppressions={Path(__file__).with_name('tsan.supp').resolve()}")
    # Fixed options prevent inherited suppressions/report disabling from hiding bugs.
    environment["TSAN_OPTIONS"] = ":".join(options)
    deadline = time.monotonic() + 45
    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = subprocess.run(command, env=environment, capture_output=True,
                                    text=True, timeout=min(20, remaining))
        except subprocess.TimeoutExpired as error:
            print(f"TSAN test timed out: {error}", file=sys.stderr)
            return 1
        if is_layout_failure(result):
            print(f"TSAN startup address conflict ({attempt}/{max_attempts}): "
                  f"{result.stderr.strip()}", file=sys.stderr, flush=True)
            continue
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if expected_warning:
            warning = f"WARNING: ThreadSanitizer: {expected_warning}"
            if result.returncode == 66 and warning in result.stderr:
                print(f"TSAN positive control detected {expected_warning} (attempt {attempt})")
                return 0
            print("TSAN positive control did not produce the intended report", file=sys.stderr)
            return 1
        if result.returncode == 0:
            print(f"TSAN test completed without unsuppressed reports (attempt {attempt})")
        return result.returncode if result.returncode >= 0 else 128 - result.returncode
    print("TSAN could not initialize within the startup attempt/time limit", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-warning", choices=("data race", "lock-order-inversion"))
    parser.add_argument("--no-gdal-suppression", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command:
        parser.error("an instrumented executable is required after --")
    return run(args.command, args.expect_warning, not args.no_gdal_suppression)


if __name__ == "__main__":
    sys.exit(main())
