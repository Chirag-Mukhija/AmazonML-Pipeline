"""Build <team>_submission.zip in the structure the organizers require:

  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md

Run from student_resource/:
  python code/business_entity_resolution/src/package.py --team <name>
"""
import argparse
import os
import zipfile

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # business_entity_resolution/
SKIP_DIRS = {".venv", "__pycache__", "work", "models", ".ipynb_checkpoints"}


def build(team: str, resource_dir: str, out_path: str):
    required = [
        os.path.join(resource_dir, "output", "matching_results.tsv"),
        os.path.join(resource_dir, "output", "candidate_pairs.tsv"),
        os.path.join(resource_dir, "Documentation_template.md"),
        os.path.join(CODE_ROOT, "README.md"),
        os.path.join(CODE_ROOT, "requirements.txt"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(resource_dir, "output", name), f"output/{name}")
        z.write(os.path.join(resource_dir, "Documentation_template.md"), "Documentation_template.md")
        for root, dirs, files in os.walk(CODE_ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in files:
                if fn.endswith((".pyc", ".DS_Store")):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, CODE_ROOT)
                z.write(full, f"code/business_entity_resolution/{rel}")
    print(f"wrote {out_path}")
    with zipfile.ZipFile(out_path) as z:
        for n in sorted(z.namelist()):
            print("  ", n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True)
    ap.add_argument("--resource-dir", default=".", help="the student_resource/ folder")
    args = ap.parse_args()
    build(args.team, args.resource_dir, os.path.join(args.resource_dir, f"{args.team}_submission.zip"))


if __name__ == "__main__":
    main()
