from __future__ import annotations

import argparse
import json
import sys

from compete.graph import run_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Competitive intelligence & market gap agent",
    )
    parser.add_argument("--company", required=True, help="Target company name")
    parser.add_argument("--description", required=True, help="One-paragraph product description")
    parser.add_argument(
        "--competitor",
        action="append",
        dest="competitors",
        required=True,
        help="Seed competitor (repeat flag for multiple)",
    )
    parser.add_argument("--run-dir", default=None, help="Optional output directory")
    args = parser.parse_args(argv)

    result = run_agent(
        company_name=args.company,
        description=args.description,
        seed_competitors=args.competitors,
        run_dir=args.run_dir,
    )

    summary = {
        "run_id": result.get("run_id"),
        "run_dir": result.get("run_dir"),
        "competitors": [c.name for c in (result.get("competitors") or [])],
        "documents": len(result.get("documents") or []),
        "skips": len(result.get("skips") or []),
        "evidence_ids": result.get("evidence_ids") or [],
        "brief_path": result.get("brief_path"),
        "evidence_json_path": result.get("evidence_json_path"),
        "errors": result.get("errors") or [],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
