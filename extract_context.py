#!/usr/bin/env python3
"""Extract all task environment archives into task_envs/{task_id}/extracted/."""

import glob
import os
import zipfile


def extract_all(task_envs_dir: str = "task_envs") -> dict[str, str]:
    """Unzip every environment archive. Returns {task_id: extract_dir}."""
    archives = glob.glob(os.path.join(task_envs_dir, "*/environment.tar.gz"))
    results = {}
    skipped = 0
    extracted = 0

    for archive_path in sorted(archives):
        task_dir = os.path.dirname(archive_path)
        task_id = os.path.basename(task_dir)
        extract_dir = os.path.join(task_dir, "extracted")

        if os.path.isdir(extract_dir) and any(os.scandir(extract_dir)):
            results[task_id] = extract_dir
            skipped += 1
            continue

        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)
            results[task_id] = extract_dir
            extracted += 1
        except (zipfile.BadZipFile, Exception) as e:
            print(f"  FAILED {task_id}: {e}")

    print(f"Extraction complete: {extracted} extracted, {skipped} already existed, {len(archives)} total")
    return results


if __name__ == "__main__":
    extract_all()
