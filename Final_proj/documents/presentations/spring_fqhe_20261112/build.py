"""Build, validate, render and package the symmetry-augmented SPRING deck.

Compilation is self-contained once the referenced PDFs are bundled. External
research inputs are hash-checked when available; absent inputs are reported as
unavailable, never as verified. No training, ED or figure regeneration occurs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
SUPPORT = HERE / 'support'
FINAL = next((path for path in HERE.parents
              if (path / 'SPRING_Haldane_FQHE').is_dir()), HERE)
EXPECTED_MAIN = 20
EXPECTED_BACKUP = 5
EXPECTED_PAGES = EXPECTED_MAIN + EXPECTED_BACKUP
RENDER = BUILD / 'render' / 'symmetry_20261116'
COMPARISON = SUPPORT / 'constraint_comparison_20261116.json'
FIGURE_RECEIPT = SUPPORT / 'constraint_figures_metadata_20261116.json'
SPHERE_RECEIPT = SUPPORT / 'figures/symmetry_sphere_metadata.json'
DEFAULT_INSPECTION = SUPPORT / 'visual_inspection_20261116.json'
ISING_SOURCE = 'SPRING_Ising_test/results/spring_n6/training.csv'
ISING_SOURCE_SHA = '791abce475ea24be5d01f7862547954d0e606ce8974c45abafe31abd9f0cebd7'
ISING_FIGURE_SHA = 'c53a2e1431560a601cb75dc3d65068eb3da8f3654b01b854a2db9721961c6868'
CURRENT_SUPPORT = (
    'constraint_comparison_20261116.json',
    'extract_constraint_comparison_20261116.py',
    'generate_constraint_figures_20261116.py',
    'constraint_figures_metadata_20261116.json',
    'constraint_theory_review_20261116.md',
    'slides_comparison_audit_20261116.md',
    'build_symmetry_sphere_20261116.py',
    'figures/symmetry_sphere_metadata.json',
)
OPTIONAL_AUDITS = (
    'visual_inspection_20261116.json',
    'revision_audit_20261116.json',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def used_graphics(source):
    """Resolve only graphics referenced by the current main TeX source."""
    names = re.findall(r'\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]+)\}', source)
    found = []
    for name in names:
        candidates = [HERE / name, SUPPORT / 'figures' / name]
        if not Path(name).suffix:
            candidates = [p.with_suffix(ext) for p in candidates
                          for ext in ('.pdf', '.png', '.jpg', '.jpeg')]
        matches = [p.resolve() for p in candidates if p.is_file()]
        if not matches:
            raise FileNotFoundError('Missing referenced graphic: ' + name)
        path = matches[0]
        if HERE not in path.parents:
            raise ValueError('A portable deck cannot reference an external graphic: ' + name)
        if path not in found:
            found.append(path)
    return found


def external_input_path(name):
    path = Path(name)
    return path if path.is_absolute() else FINAL / path


def verify_external_inputs(records):
    verified, unavailable = {}, {}
    for name, expected in records.items():
        path = external_input_path(name)
        if path.is_file():
            actual = digest(path)
            if actual != expected:
                raise ValueError('External input hash mismatch: ' + name)
            verified[name] = expected
        else:
            unavailable[name] = expected
    return dict(total=len(records), verified_count=len(verified),
                unavailable_count=len(unavailable), all_available=not unavailable,
                all_available_hashes_match=True, verified_sha256=verified,
                unavailable_sha256=unavailable,
                scope='Missing originals do not prevent a self-contained PDF rebuild; no missing input is counted as verified')


def receipt_output_hash(record, asset):
    values = record.get('file_sha256', record.get('artifacts', {}))
    matches = [value for name, value in values.items()
               if Path(name).name == asset.name]
    if len(matches) != 1:
        raise ValueError('Receipt does not uniquely bind used asset: ' + asset.name)
    return matches[0]


def verify_local_receipts(graphics, comparison):
    figures = read_json(FIGURE_RECEIPT)
    if figures.get('status') != 'passed':
        raise ValueError('Current numerical figure receipt has not passed')
    generator = SUPPORT / 'generate_constraint_figures_20261116.py'
    if digest(generator) != figures['generator_sha256']:
        raise ValueError('Current numerical figure generator hash mismatch')
    if digest(COMPARISON) != figures['manifest_sha256']:
        raise ValueError('Figure manifest hash mismatch')
    extractor = SUPPORT / 'extract_constraint_comparison_20261116.py'
    if digest(extractor) != comparison['extraction_script_sha256']:
        raise ValueError('Comparison extraction script hash mismatch')
    sphere = read_json(SPHERE_RECEIPT)
    if sphere.get('sphere_has_no_boundary') is not True or sphere.get('projected_edge_potential_present') is not False:
        raise ValueError('Sphere schematic does not match the unconfined model')
    if digest(SUPPORT / 'build_symmetry_sphere_20261116.py') != sphere['script_sha256']:
        raise ValueError('Sphere generator hash mismatch')
    outputs = {}
    for asset in graphics:
        if asset.name.startswith('constraint_'):
            expected = receipt_output_hash(figures, asset)
        elif asset.name == 'haldane_sphere_symmetry.pdf':
            expected = receipt_output_hash(sphere, asset)
        elif asset.name == 'slide_ising.pdf':
            # This retained PDF is bound below; only its actual Ising source is
            # checked, not the unused historical wall/depth inputs.
            expected = ISING_FIGURE_SHA
        else:
            raise ValueError('Add explicit provenance for the new referenced asset: ' + asset.name)
        if digest(asset) != expected:
            raise ValueError('Used graphic hash mismatch: ' + asset.name)
        outputs[str(asset.relative_to(HERE))] = expected
    return dict(status='passed', used_graphics_sha256=outputs,
                numerical_manifest_sha256=digest(COMPARISON),
                numerical_figure_receipt_sha256=digest(FIGURE_RECEIPT),
                sphere_receipt_sha256=digest(SPHERE_RECEIPT),
                only_main_referenced_graphics_required=True)


def pdf_text_checks(extracted):
    compact = re.sub(r'\s+', '', extracted)
    lower = extracted.lower()
    checks = {
        'Haldane sphere': 'Haldane sphere' in extracted,
        'SPRING': 'SPRING' in extracted,
        'MinSR': 'MinSR' in extracted,
        'current_parameter_count': '294,464' in compact,
        'latest_fidelity': bool(re.search(r'77\.(?:817|82)', compact)),
        'latest_physical_energy': bool(re.search(r'6\.404(?:269|27)', compact)),
        'mixed_H_ED_target': '6.244768' in compact,
        'inherited_CG_limit': '1024' in compact,
        'commuting_constraint_discussion': 'commut' in lower,
        'mixed_pseudopotentials': '0.8' in compact and '0.6' in compact,
    }
    for name, passed in checks.items():
        if not passed:
            raise ValueError('Missing revised PDF content: ' + name)
    forbidden = {
        'edge potential': r'edge\s+potential',
        'projected wall': r'projected\s+wall',
        'old spatial cutoff': r'\bx\s*_?\s*c\b',
        'old confined ED energy': r'2\.82308',
    }
    for name, pattern in forbidden.items():
        if re.search(pattern, lower):
            raise ValueError('Removed content remains in PDF: ' + name)
    return dict(required=checks, removed_content_absent=list(forbidden))


def inspection_status(pdf, receipt):
    """A render is not a visual inspection; only a matching receipt can pass."""
    if not receipt.is_file():
        return dict(status='pending', reason='Rendered pages require visual inspection',
                    expected_receipt=str(receipt.relative_to(HERE)) if HERE in receipt.parents else str(receipt))
    record = read_json(receipt)
    valid = (record.get('status') == 'passed'
             and record.get('pdf_sha256') == digest(pdf)
             and record.get('pages_inspected') == list(range(1, EXPECTED_PAGES + 1)))
    if not valid:
        return dict(status='pending', reason='Inspection receipt is incomplete or belongs to a different PDF',
                    receipt_sha256=digest(receipt))
    return dict(status='passed', receipt_sha256=digest(receipt),
                pages_inspected=record['pages_inspected'], pdf_sha256=record['pdf_sha256'])


def package(validation):
    """Package the current deck, current support, and used graphics only."""
    paths = [HERE / name for name in
             ('SPRING_FQHE.tex', 'SPRING_FQHE.pdf', 'README.md', 'build.py', 'validation.json')]
    paths += [HERE / name for name in validation['local_asset_verification']['used_graphics_sha256']]
    paths += [SUPPORT / name for name in CURRENT_SUPPORT]
    paths += [SUPPORT / name for name in OPTIONAL_AUDITS if (SUPPORT / name).is_file()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing current package files: ' + repr(missing))
    unique = sorted(set(paths))
    with zipfile.ZipFile(HERE / 'SPRING_FQHE_source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique:
            archive.write(path, path.relative_to(HERE))
    return [str(path.relative_to(HERE)) for path in unique]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--require-external-inputs', action='store_true',
                        help='Fail if the original research input files are unavailable')
    parser.add_argument('--inspection-receipt', type=Path, default=DEFAULT_INSPECTION,
                        help='JSON with status passed, current pdf_sha256 and pages_inspected 1..25')
    args = parser.parse_args()
    BUILD.mkdir(exist_ok=True)
    source = (HERE / 'SPRING_FQHE.tex').read_text()
    parts = source.split('\\appendix', 1)
    if len(parts) != 2:
        raise ValueError('The revised deck needs a main/backup boundary')
    main_frames = len(re.findall(r'\\begin\{frame\}', parts[0]))
    backup_frames = len(re.findall(r'\\begin\{frame\}', parts[1]))
    if (main_frames, backup_frames) != (EXPECTED_MAIN, EXPECTED_BACKUP):
        raise ValueError('Expected 20 main + 5 backup frames, got ' + repr((main_frames, backup_frames)))
    graphics = used_graphics(source)
    comparison = read_json(COMPARISON)
    if comparison.get('status') != 'passed' or not comparison.get('no_training_or_ED_performed'):
        raise ValueError('Current saved-result extraction has not passed')
    system = comparison['system']
    if not (system['Ne'] == 7 and system['dimension'] == 1656
            and system['edge_potential'] is False and system['v3'] == .8 and system['v5'] == .6):
        raise ValueError('Wrong physical model in comparison manifest')
    if abs(system['ED_energy0_V1'] - 6.2447682602150385) > 1e-12:
        raise ValueError('Wrong mixed-H ED target')
    local = verify_local_receipts(graphics, comparison)
    external = verify_external_inputs(comparison['input_sha256'])
    ising = None
    if any(path.name == 'slide_ising.pdf' for path in graphics):
        ising = verify_external_inputs({ISING_SOURCE: ISING_SOURCE_SHA})
    if args.require_external_inputs and (not external['all_available']
                                        or (ising is not None and not ising['all_available'])):
        raise RuntimeError('Original research files unavailable under --require-external-inputs')
    with (BUILD / 'build.log').open('w') as log:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                        '-outdir=build', 'SPRING_FQHE.tex'], cwd=HERE,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    pdf = BUILD / 'SPRING_FQHE.pdf'
    log = (BUILD / 'SPRING_FQHE.log').read_text()
    issues = re.findall(r'^.*(?:Overfull|Underfull|Warning:|Error:).*$', log, re.M)
    if issues:
        raise RuntimeError('Resolve layout/build warnings: ' + repr(issues))
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    if pages != EXPECTED_PAGES:
        raise ValueError('Expected 25 PDF pages, got ' + str(pages))
    text_path = BUILD / 'SPRING_FQHE.txt'
    subprocess.run(['pdftotext', '-layout', str(pdf), str(text_path)], check=True)
    extracted = text_path.read_text()
    if len(extracted.split('\f')) - 1 != pages:
        raise ValueError('PDF text/page segmentation mismatch')
    content = pdf_text_checks(extracted)
    shutil.copyfile(pdf, HERE / pdf.name)
    if args.render:
        RENDER.mkdir(parents=True, exist_ok=True)
        subprocess.run(['pdftoppm', '-r', '125', '-png', str(pdf), str(RENDER / 'slide')], check=True)
        if len(list(RENDER.glob('slide-*.png'))) != pages:
            raise ValueError('Rendered page count mismatch')
    previous = read_json(HERE / 'validation.json') if (HERE / 'validation.json').is_file() else {}
    previous_render_current = (previous.get('pdf_sha256') == digest(HERE / pdf.name)
                               and previous.get('all_pages_rendered') is True)
    rendered = args.render or previous_render_current
    validation = dict(
        status='passed', date='2026-11-16', main_slides=EXPECTED_MAIN,
        backup_slides=EXPECTED_BACKUP, total_pages=pages, aspect_ratio='16:9',
        build_warnings=issues, pdf_sha256=digest(HERE / pdf.name),
        source_sha256=digest(HERE / 'SPRING_FQHE.tex'), pdf_content_checks=content,
        local_asset_verification=local, external_research_inputs=external,
        retained_Ising_source=ising,
        all_external_inputs_available=external['all_available'] and (ising is None or ising['all_available']),
        all_pages_rendered=rendered,
        render_directory=str(RENDER.relative_to(HERE)) if rendered else None,
        visual_inspection=inspection_status(HERE / pdf.name, args.inspection_receipt.resolve()),
        no_new_training=True, no_new_ED=True,
        scientific_reviews=['support/constraint_theory_review_20261116.md',
                            'support/slides_comparison_audit_20261116.md'],
        comparison_manifest_sha256=digest(COMPARISON),
        figure_regeneration_scope='Current constraint and sphere generators use bundled scalar data; retained Ising regeneration requires its original trace',
        portable_rebuild_scope='Bundled vector assets suffice for compilation; unavailable original research files are explicitly listed',
        scope='Revised commuting-constraint presentation of retained calculations; no removed confinement figures packaged',
    )
    (HERE / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    validation['package_files'] = package(validation)
    (HERE / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    # Repackage once so the archive contains the final inventory-bearing receipt.
    package(validation)
    print(f'Passed: {pages} slides; {external["verified_count"]}/{external["total"]} original comparison inputs verified; clean LaTeX build.')
    print('Visual inspection: ' + validation['visual_inspection']['status'])
    print(HERE / 'SPRING_FQHE.pdf')


if __name__ == '__main__':
    main()
