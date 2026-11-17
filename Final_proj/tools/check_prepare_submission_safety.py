#!/usr/bin/env python3
"""Exercise exporter safeguards in /tmp; no real Git process or network is used."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import prepare_submission as export
import validate_reorganization as inventory


def checks():
    passed = []
    with tempfile.TemporaryDirectory(prefix='applied_ml_export_checks_', dir='/tmp') as temporary:
        base = Path(temporary)
        source, destination = base / 'Final_proj', base / '.submission' / 'applied-ml'
        source.mkdir()
        for directory in inventory.REQUIRED_DIRECTORIES:
            (source / directory).mkdir()
            (source / directory / 'README.md').write_text('# Component\n')
        for name in ('README.md', 'SUBMISSION.md', 'THIRD_PARTY.md'):
            (source / name).write_text('# Current document\n')
        (source / 'MIGRATION.md').write_text('[Archived only](SPRING_Haldane_FQHE/results/old/README.md)\n')
        legacy = source / 'SPRING_Haldane_FQHE'
        legacy.mkdir()
        (legacy / 'untouched.txt').write_text('retain local research\n')
        (source / 'benchmarks' / 'small_reference.npz').write_bytes(b'NPZ fixture, not loaded')
        (source / 'SPRING/results/runs/fresh').mkdir(parents=True)
        (source / 'SPRING/results/runs/fresh/output.json').write_text('{}')
        (source / 'SPRING/results/check_cpu_n4').mkdir(parents=True)
        (source / 'SPRING/results/check_cpu_n4/metrics.csv').write_text('energy\n0\n')
        historical = source / 'shared' / 'snapshot'
        historical.mkdir()
        (historical / 'README.md').write_text('[Frozen historical link](missing_old_file.py)\n')
        outside = base / 'outside'
        outside.mkdir()
        (outside / 'not_exported.txt').write_text('not followed\n')
        (source / 'shared' / 'symlink_dir').symlink_to(outside, target_is_directory=True)
        found, catalog, report = inventory.build_metadata(source)
        assert not report['errors'], report['errors']
        assert 'benchmarks/small_reference.npz' in found['files']
        assert 'SPRING/results/check_cpu_n4/metrics.csv' in found['files']
        assert not any('/results/runs/' in name for name in found['files'])
        assert not any('symlink_dir' in name for name in found['files'])
        assert len(report['links']['permitted_archive_index_links']) == 1
        assert len(report['links']['historical_unresolved_links']) == 1
        assert str(base) not in inventory.json_bytes(catalog).decode()
        passed += ['small_NPZ_retained', 'symlink_directory_not_followed',
                   'fresh_results_runs_excluded_but_fixed_evidence_retained',
                   'legacy_exception_only_in_MIGRATION', 'historical_links_nonblocking', 'catalog_paths_relative']
        (source / 'SPRING' / 'README.md').write_text('[Not current](../SPRING_Haldane_FQHE/old.md)\n')
        assert inventory.build_metadata(source)[2]['errors']
        (source / 'SPRING' / 'README.md').write_text('# Component\n')
        passed.append('legacy_link_in_current_algorithm_doc_rejected')
        large = source / 'benchmarks' / 'large.npz'
        with large.open('wb') as stream:
            stream.truncate(inventory.MAX_FILE_BYTES)
        assert any(q.get('bytes') == inventory.MAX_FILE_BYTES for q in inventory.collect(source)['errors'])
        large.unlink()
        private = source / 'shared' / 'credentials.json'
        private.write_text('{}')
        assert any(q['path'] == 'shared/credentials.json' for q in inventory.collect(source)['errors'])
        private.unlink()
        passed += ['file_at_50MB_refused', 'private_path_refused']
        try:
            export.inspect_destination(source, destination)
            raise AssertionError('Missing destination was accepted')
        except export.PreparationError:
            passed.append('missing_destination_refused')
        destination.mkdir(parents=True)
        with patch.object(export, 'git', return_value='/home/leean/research') as fake:
            try:
                export.inspect_destination(source, destination)
                raise AssertionError('Research-container Git root was accepted')
            except export.PreparationError:
                assert fake.call_args.args[1:] == ('rev-parse', '--show-toplevel')
                assert fake.call_count == 1
        passed.append('research_container_Git_root_refused_before_other_Git_inspection')
        with patch.object(export, 'git') as fake:
            try:
                export.inspect_destination(source, base / 'wrong')
                raise AssertionError('Wrong destination was accepted')
            except export.PreparationError:
                fake.assert_not_called()
        passed.append('wrong_destination_refused_without_Git_call')
        (destination / '.git').mkdir()
        (destination / '.git' / 'sentinel').write_text('Git metadata unchanged\n')
        (destination / 'README.md').write_text('old root landing\n')
        (destination / '.gitignore').write_text('__pycache__/\n')
        (destination / 'unrelated.txt').write_text('preserve other repository files\n')
        (destination / 'Final_proj').mkdir()
        (destination / 'Final_proj' / 'old.txt').write_text('old payload\n')

        def local_git(repo, *arguments):
            if arguments == ('rev-parse', '--show-toplevel'):
                return str(destination.resolve())
            if arguments == ('remote', 'get-url', 'origin'):
                return export.REPOSITORY + '.git'
            if arguments == ('rev-parse', 'HEAD'):
                return 'a' * 40
            if arguments == ('branch', '--show-current'):
                return 'main'
            if arguments[0] == 'status':
                return ''
            raise AssertionError('Unexpected Git operation: ' + repr(arguments))

        with patch.object(export, 'git', side_effect=local_git):
            prepared = export.preview(source, destination, 'MIGRATION.md', True, '2026-11-17')
            assert not prepared['errors'], prepared['errors']
            result = export.apply(prepared)
            assert (Path(result['backup_directory']) / 'Final_proj' / 'old.txt').read_text() == 'old payload\n'
            assert not (destination / 'Final_proj' / 'old.txt').exists()
            assert not (destination / 'Final_proj' / 'SPRING_Haldane_FQHE').exists()
            assert (destination / 'unrelated.txt').read_text() == 'preserve other repository files\n'
            assert (destination / '.git' / 'sentinel').read_text() == 'Git metadata unchanged\n'
            assert (legacy / 'untouched.txt').read_text() == 'retain local research\n'
            passed += ['verified_curated_swap_and_old_payload_backup', 'unrelated_files_and_Git_metadata_preserved',
                       'source_legacy_research_preserved', 'manifest_hashes_verified']
            before = {name: inventory.sha256(destination / name) for name in ('README.md', 'SUBMISSION_MANIFEST.json')}
            prepared = export.preview(source, destination, 'MIGRATION.md', True, '2026-11-17')
            original = (source / 'SPRING' / 'README.md').read_bytes()
            (source / 'SPRING' / 'README.md').write_text('changed after preview\n')
            try:
                export.apply(prepared)
                raise AssertionError('Changed source was accepted')
            except export.PreparationError:
                assert all(inventory.sha256(destination / name) == digest for name, digest in before.items())
            (source / 'SPRING' / 'README.md').write_bytes(original)
            passed.append('source_hash_change_refused_before_swap')
            prepared = export.preview(source, destination, 'MIGRATION.md', True, '2026-11-17')
            original_atomic = export.atomic_write

            def fail_manifest(path, content):
                if path.name == 'SUBMISSION_MANIFEST.json':
                    raise OSError('intentional scratch-only write failure')
                return original_atomic(path, content)

            with patch.object(export, 'atomic_write', side_effect=fail_manifest):
                try:
                    export.apply(prepared)
                    raise AssertionError('Injected transaction failure did not propagate')
                except OSError:
                    assert all(inventory.sha256(destination / name) == digest for name, digest in before.items())
                    export.verify_payload(destination / 'Final_proj', prepared['payload'])
            passed.append('transaction_failure_rolls_back_payload_and_owned_metadata')
    return {'status': 'passed', 'scratch_scope': '/tmp only; temporary fake-Git fixture removed',
            'real_Git_processes_or_network_used': False, 'actual_submission_checkout_modified': False,
            'checks': passed, 'helper_sha256': {name: inventory.sha256(Path(__file__).with_name(name)) for name in
                ('prepare_submission.py', 'validate_reorganization.py', 'check_prepare_submission_safety.py')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = checks()
    if args.receipt:
        inventory.atomic_write(args.receipt, inventory.json_bytes(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
