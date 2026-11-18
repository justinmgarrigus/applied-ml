#!/usr/bin/env python3
"""Prepare a curated working-tree export; never stage, commit, push or use a network.

Default: --check/--dry-run, which writes nothing. --apply requires the exact
project .submission/applied-ml Git boundary and a clean working tree. It builds
and verifies a temporary payload before swapping only Final_proj, retaining a
recoverable backup outside the repository. Use --refresh-root-readme explicitly
to replace the old repository landing page with current algorithm navigation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import tempfile
import uuid
from urllib.parse import unquote, urlsplit

from validate_reorganization import (GENERATED, LEGACY_ROOTS, _markdown_destinations,
    atomic_write, build_metadata, json_bytes, sha256)

REPOSITORY = 'https://github.com/justinmgarrigus/applied-ml'
EXPECTED_REMOTES = {REPOSITORY, REPOSITORY + '.git', 'git@github.com:justinmgarrigus/applied-ml.git'}


class PreparationError(RuntimeError):
    pass


def git(destination: Path, *args) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(['git', '-C', str(destination), *args], env=env,
                            text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise PreparationError('Read-only Git inspection failed: ' + ' '.join(args))
    return result.stdout.strip()


def inspect_destination(source: Path, destination: Path) -> dict:
    expected = source.parent / '.submission' / 'applied-ml'
    if destination.absolute() != expected.absolute():
        raise PreparationError('Destination must be exactly the project .submission/applied-ml checkout')
    if destination.is_symlink() or destination.parent.is_symlink():
        raise PreparationError('Destination and its .submission parent must not be symlinks')
    if not destination.is_dir():
        raise PreparationError('The intended submission checkout does not exist')
    # The Git boundary check precedes every other Git inspection or mutation.
    top = Path(git(destination, 'rev-parse', '--show-toplevel')).resolve()
    if top == Path('/home/leean/research') or top == source.parent.parent:
        raise PreparationError('STOP: Git resolved to the research container, not the submission project')
    if top != destination.resolve():
        raise PreparationError('STOP: destination is not its own exact Git repository boundary')
    if git(destination, 'remote', 'get-url', 'origin') not in EXPECTED_REMOTES:
        raise PreparationError('The destination origin is not the intended applied-ml repository')
    if (destination / 'Final_proj').is_symlink():
        raise PreparationError('The existing Final_proj payload must not be a symlink')
    return {'boundary': '.submission/applied-ml', 'HEAD': git(destination, 'rev-parse', 'HEAD'),
            'branch': git(destination, 'branch', '--show-current'),
            'status': git(destination, 'status', '--porcelain=v1', '--untracked-files=all')}


def landing_readme() -> bytes:
    return b'''# Applied ML final project

Contributor: [PassengerAn](https://github.com/PassengerAn).

The project is organized by optimization method. Start with SPRING, validate
against the available small-system benchmarks, then inspect the VMC and MinSR
comparison modules. Each module documents its implemented scope and limitations.

- [Project overview and current results](Final_proj/README.md)
- [SPRING implementation and validation](Final_proj/SPRING/)
- [VMC module](Final_proj/VMC/)
- [MinSR module](Final_proj/MinSR/)
- [Shared models and pinned source snapshot](Final_proj/shared/)
- [Benchmark data](Final_proj/benchmarks/)
- [Course submission guide](Final_proj/SUBMISSION.md)
- [Third-party attribution](Final_proj/THIRD_PARTY.md)
- [Migration and local legacy archive index](Final_proj/MIGRATION.md)
- [Curated source catalog](Final_proj/SOURCE_CATALOG.json)
- [Submission manifest](SUBMISSION_MANIFEST.json)

Raw training archives, large checkpoints and compilation caches remain outside
this curated submission. The written course report is submitted separately
through Canvas as described in the submission guide.
'''


def check_root_readme(content: bytes, payload_paths: set[str], destination: Path) -> list[dict]:
    available = {'Final_proj/' + q for q in payload_paths} | {'README.md', '.gitignore', 'SUBMISSION_MANIFEST.json'}
    directories = set()
    for name in available:
        directories.update(str(p) for p in PurePosixPath(name).parents)
    errors = []
    for target in _markdown_destinations(content.decode(errors='replace')):
        if not target or target.startswith('#') or urlsplit(target).scheme or target.startswith('//'):
            continue
        normalized = posixpath.normpath(unquote(target.split('#', 1)[0].split('?', 1)[0].strip('<>')))
        if normalized in available or normalized in directories:
            continue
        existing = destination / normalized
        if not normalized.startswith(('..', '/', 'Final_proj/')) and existing.is_file() and not existing.is_symlink():
            continue
        errors.append({'document': 'repository README.md', 'target': target,
                       'reason': 'not in the prospective curated package; use --refresh-root-readme or edit it first'})
    return errors


def preview(source: Path, destination: Path, legacy_index: str, refresh_root_readme: bool,
            snapshot_date: str) -> dict:
    inventory, catalog, validation = build_metadata(source, legacy_index)
    state = inspect_destination(source, destination)
    root_readme = landing_readme() if refresh_root_readme else (destination / 'README.md').read_bytes()
    metadata = {GENERATED[0]: json_bytes(catalog), GENERATED[1]: json_bytes(validation)}
    payload = dict(inventory['files'])
    for name, data in metadata.items():
        payload[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    errors = list(validation['errors']) + check_root_readme(root_readme, set(payload), destination)
    files = {'Final_proj/' + name: item for name, item in sorted(payload.items())}
    files['README.md'] = {'bytes': len(root_readme), 'sha256': hashlib.sha256(root_readme).hexdigest()}
    ignore = destination / '.gitignore'
    if ignore.is_file():
        files['.gitignore'] = {'bytes': ignore.stat().st_size, 'sha256': sha256(ignore)}
    manifest = {'snapshot_date': snapshot_date, 'contributor_handle': 'PassengerAn', 'repository': REPOSITORY,
        'scope': 'Curated algorithm-first package; unrelated repository files are preserved and outside this manifest',
        'file_count_excluding_manifest': len(files), 'bytes_excluding_manifest': sum(q['bytes'] for q in files.values()),
        'excluded_local_content': ['legacy raw research trees', 'virtual environments and compilation caches',
                                   'symlinks', 'secret/private paths', 'files >=50 MB'],
        'files': dict(sorted(files.items()))}
    return {'source': source, 'destination': destination, 'state': state, 'inventory': inventory,
            'metadata': metadata, 'payload': payload, 'root_readme': root_readme,
            'refresh_root_readme': refresh_root_readme, 'manifest': manifest, 'errors': errors,
            'validation': validation}


def verify_payload(root: Path, records: dict) -> None:
    actual = set()
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if (Path(folder) / name).is_symlink():
                raise PreparationError('A symlink appeared in the verified temporary payload')
        for name in files:
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or relative not in records or sha256(path) != records[relative]['sha256']:
                raise PreparationError('Temporary/exported payload hash mismatch: ' + relative)
            actual.add(relative)
    if actual != set(records):
        raise PreparationError('Temporary/exported payload inventory mismatch')


def apply(prepared: dict) -> dict:
    source, destination = prepared['source'], prepared['destination']
    if prepared['errors']:
        raise PreparationError('Refusing export: curated validation has errors')
    if prepared['state']['status']:
        raise PreparationError('Refusing export: submission checkout is not clean; preserve/review its existing edits first')
    staging = Path(tempfile.mkdtemp(prefix='.curated_export_', dir=destination.parent))
    stage_payload = staging / 'Final_proj'
    stage_payload.mkdir()
    backup = None
    moved_old = installed = False
    old_metadata = {}
    try:
        for relative, item in prepared['inventory']['files'].items():
            original, target = source / relative, stage_payload / relative
            if original.is_symlink():
                raise PreparationError('Source became a symlink during export: ' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            if sha256(target) != item['sha256']:
                raise PreparationError('Source changed during export; run --check again: ' + relative)
        for relative, data in prepared['metadata'].items():
            atomic_write(stage_payload / relative, data)
        verify_payload(stage_payload, prepared['payload'])
        current = inspect_destination(source, destination)
        if current != prepared['state']:
            raise PreparationError('Submission Git HEAD/branch/worktree changed during preparation')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        backup = destination.parent / 'prepared_backups' / stamp
        backup.mkdir(parents=True, exist_ok=False)
        for name in ('SUBMISSION_MANIFEST.json', 'README.md'):
            path = destination / name
            old_metadata[name] = path.read_bytes() if path.exists() else None
            if path.exists():
                shutil.copy2(path, backup / name)
        if (destination / 'Final_proj').exists():
            (destination / 'Final_proj').replace(backup / 'Final_proj')
            moved_old = True
        stage_payload.replace(destination / 'Final_proj')
        installed = True
        if prepared['refresh_root_readme']:
            atomic_write(destination / 'README.md', prepared['root_readme'])
        atomic_write(destination / 'SUBMISSION_MANIFEST.json', json_bytes(prepared['manifest']))
        verify_payload(destination / 'Final_proj', prepared['payload'])
        for name, record in prepared['manifest']['files'].items():
            if sha256(destination / name) != record['sha256']:
                raise PreparationError('Final submission manifest verification failed: ' + name)
        receipt = {'status': 'prepared_working_tree_only', 'repository': REPOSITORY,
            'previous_HEAD': prepared['state']['HEAD'], 'branch': prepared['state']['branch'],
            'file_count_excluding_manifest': prepared['manifest']['file_count_excluding_manifest'],
            'bytes_excluding_manifest': prepared['manifest']['bytes_excluding_manifest'],
            'SUBMISSION_MANIFEST_sha256': sha256(destination / 'SUBMISSION_MANIFEST.json'),
            'legacy_payload_removed_from_prepared_checkout_only': list(LEGACY_ROOTS),
            'working_source_legacy_archives_preserved': True, 'source_files_changed': False,
            'verified_temporary_payload_before_swap': True, 'old_payload_backup_retained': True,
            'Git_index_commit_refs_credentials_and_network_unchanged': True,
            'next_step': 'Review git diff/status and run checks; root decides any commit/push separately'}
        atomic_write(backup / 'preparation_receipt.json', json_bytes(receipt))
        return dict(receipt, backup_directory=str(backup))
    except BaseException:
        if installed and (destination / 'Final_proj').exists():
            (destination / 'Final_proj').replace(staging / 'failed_payload')
        if moved_old:
            (backup / 'Final_proj').replace(destination / 'Final_proj')
        for name, content in old_metadata.items():
            target = destination / name
            if content is None:
                if target.exists():
                    target.unlink()
            else:
                # Restore only the two exact metadata paths this transaction owns.
                target.write_bytes(content)
        raise
    finally:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', '--dry-run', action='store_true', help='Read-only preview (default)')
    mode.add_argument('--apply', action='store_true', help='Explicitly prepare the working-tree payload; no staging/commit/push')
    parser.add_argument('--legacy-index', default='MIGRATION.md')
    parser.add_argument('--refresh-root-readme', action='store_true', help='Explicitly replace the repository landing README')
    parser.add_argument('--snapshot-date', default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.absolute() if args.destination else source.parent / '.submission' / 'applied-ml'
    try:
        prepared = preview(source, destination, args.legacy_index, args.refresh_root_readme, args.snapshot_date)
        if args.apply:
            result = apply(prepared)
        else:
            result = {'status': 'passed' if not prepared['errors'] else 'failed', 'mode': 'read_only_dry_run',
                'files_excluding_manifest': prepared['manifest']['file_count_excluding_manifest'],
                'bytes_excluding_manifest': prepared['manifest']['bytes_excluding_manifest'],
                'Git_branch': prepared['state']['branch'], 'Git_HEAD': prepared['state']['HEAD'],
                'destination_worktree_clean': not bool(prepared['state']['status']),
                'root_readme_would_be_refreshed': args.refresh_root_readme,
                'errors': prepared['errors'], 'historical_unresolved_links': len(prepared['validation']['links']['historical_unresolved_links']),
                'no_files_or_Git_state_changed': True}
            if args.json:
                result['validation'] = prepared['validation']
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if not prepared['errors'] else 1)
    except (PreparationError, OSError, ValueError) as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
