"""Ask whether a sealed test would be accepted — without taking it.

A sealed test is a one-shot. Its firewall refuses a release that is incomplete,
that carries unplanned runs, or whose science modules have moved since the suite
ran, and that refusal arrives at the one moment it is most expensive: after
hours of compute, when the only remedy for a drifted fingerprint is to run the
suite again.

So the gates are worth asking about early and often. This module calls a
protocol's own ``_check`` and ``_eligible`` — the real ones, not copies — and
reports what they say. Neither reads the sealed split; the split is only touched
by the evaluation loop those two functions guard, and that loop is not called
here. Running this leaves the seal intact and the manifest unmarked, which is
the whole point: a reader can confirm a release is *eligible* without anyone
having seen a test number.

It also reports what the test would cover if taken now. That matters because the
firewall treats a declared extension as all-or-nothing: a partially complete
extension is refused outright, while an extension with no records at all is
silently skipped and can never be evaluated afterwards, because the manifest is
marked once. "Eligible" and "eligible for everything you meant to measure" are
different questions, and this answers both.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def preflight(version: str, out_dir: Path) -> dict:
    """Report sealed-test eligibility for one protocol. Reads no test data."""
    ft = import_module(f".{version}.finaltest", package=__package__)
    out_dir = Path(out_dir)
    report: dict = {"version": version, "out_dir": str(out_dir)}

    try:
        ft._check(out_dir)
    except ft.FinalTestRefused as exc:
        report["eligible"] = False
        report["stage"] = "check"
        report["reason"] = str(exc)
        return report
    report["checks_passed"] = True

    try:
        eligible, with_extension = ft._eligible(out_dir)
    except ft.FinalTestRefused as exc:
        report["eligible"] = False
        report["stage"] = "eligible"
        report["reason"] = str(exc)
        return report

    report["eligible"] = True
    report["n_runs"] = len(eligible)
    report["extension_included"] = with_extension
    return report


def describe(report: dict) -> str:
    v = report["version"]
    if not report["eligible"]:
        return (
            f"{v}: NOT eligible — the sealed test would be refused at the "
            f"{report['stage']} gate.\n    {report['reason']}"
        )
    lines = [
        f"{v}: eligible. Manifest, completeness and all fingerprints check out.",
        f"    the sealed test would cover {report['n_runs']} runs, "
        f"extension {'included' if report['extension_included'] else 'EXCLUDED'}.",
    ]
    if not report["extension_included"]:
        lines.append(
            "    the extension has no records, so taking the test now would "
            "forfeit it permanently."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version", help="protocol version, e.g. v6")
    ap.add_argument("--out", required=True, help="release directory")
    args = ap.parse_args(argv)
    print(describe(preflight(args.version, Path(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
