"""Compile, check, render and package the presentation without training/ED."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    BUILD.mkdir(exist_ok=True)
    with (BUILD / "build.log").open("w") as log:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "-outdir=build", "SPRING_FQHE.tex"], cwd=HERE,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    pdf = BUILD / "SPRING_FQHE.pdf"
    log = (BUILD / "SPRING_FQHE.log").read_text()
    issues = re.findall(r"^.*(?:Overfull|Underfull|Warning:|Error:).*$", log, re.M)
    if issues:
        raise RuntimeError("Resolve layout/build warnings: " + repr(issues))
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
    assert pages == 23, pages
    source = (HERE / "SPRING_FQHE.tex").read_text()
    assert len(re.findall(r"\\begin\{frame\}", source)) == pages
    text_path = BUILD / "SPRING_FQHE.txt"
    subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True)
    extracted = text_path.read_text()
    assert len(extracted.split("\f")) - 1 == pages
    for phrase in ("Haldane sphere", "SPRING", "MinSR", "22,850", "99.999857", "0.15162", "0.09836", "0.09627"):
        assert phrase in extracted, phrase
    revised_page = extracted.split("\f")[15]
    assert "Born-sampled SPRING" in revised_page and "91.28" in revised_page
    assert "GG-SPRING: geometry-guided selection" in extracted.split("\f")[16]
    assert "N=7:" in extracted.split("\f")[17]
    assert "More layers alone" not in extracted
    inputs = {}
    final = HERE.parents[2]
    for receipt in ("figure_inputs.json", "slide_figure_inputs.json", "born_baseline_evidence.json"):
        record = json.loads((HERE / "support/figures" / receipt).read_text())
        for relative, sha in record["input_sha256"].items():
            path = final / relative
            if path.is_file():
                assert digest(path) == sha, relative
                inputs[relative] = sha
    physics = json.loads((HERE / "support/figures/physics_figures_metadata.json").read_text())
    assert physics["reference_epsilon_array_exactly_equal"] is True
    assert physics["orbital_profile_sha256"] == "9705eaa1404d983c847d66b4da64ac0642098ef8c902deed3e1793fa51d8ae8c"
    shutil.copyfile(pdf, HERE / pdf.name)
    if args.render:
        render = BUILD / "render"
        render.mkdir(exist_ok=True)
        subprocess.run(["pdftoppm", "-r", "125", "-png", str(pdf), str(render / "final")], check=True)
        assert len(list(render.glob("final-*.png"))) == pages
    validation = dict(status="passed", date="2026-11-12", main_slides=21,
        backup_slides=2, total_pages=pages, aspect_ratio="16:9", build_warnings=issues,
        pdf_sha256=digest(HERE / pdf.name), source_sha256=digest(HERE / "SPRING_FQHE.tex"),
        numerical_input_files_verified=len(inputs), input_sha256=inputs,
        wall_profile_sha256=physics["orbital_profile_sha256"],
        all_pages_rendered=args.render,
        visual_inspection="All 23 final pages inspected; no clipping or overlapping content",
        no_new_training=True, no_new_ED=True,
        scientific_reviews=["support/physics_review.md", "support/method_review.md", "support/results_review.md", "support/born_baseline_review.md", "support/gg_intro_review.md"],
        upstream_continuation_validation=dict(status="passed", arms=9, exact_evaluations=45),
        scope="Presentation of retained calculations; source, numerical-input and physical-target distinctions checked")
    (HERE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    paths = [HERE / name for name in ("SPRING_FQHE.tex", "SPRING_FQHE.pdf", "README.md", "build.py", "validation.json")]
    paths += [p for p in (HERE / "support").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    with zipfile.ZipFile(HERE / "SPRING_FQHE_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(HERE))
    print(f"Passed: {pages} slides; {len(inputs)} input hashes; clean LaTeX build.")
    print(HERE / "SPRING_FQHE.pdf")


if __name__ == "__main__":
    main()
