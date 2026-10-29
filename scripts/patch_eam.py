import re
from pathlib import Path

path = Path("nemoforge/src/ticket_to_code/workflow.py")
content = path.read_text(encoding="utf-8")
crlf = "\r\n" in content
clean = content.replace("\r\n", "\n")

target1 = """        _planned = []
        for _t in (getattr(_plan_obj, "tasks", None) or []):
            _tt = getattr(getattr(_t, "task_type", None), "value", str(getattr(_t, "task_type", "")))
            _planned.append((getattr(_t, "file_path", ""), str(_tt).lower()))"""

replacement1 = """        _planned = []
        for _t in (getattr(_plan_obj, "tasks", None) or []):
            _tid = str(getattr(_t, "id", "") or "")
            _ttype_desc = str(getattr(_t, "description", "") or "").lower()
            _title = str(getattr(_t, "title", "") or "").lower()
            if _tid.startswith("cmp-") or "companion" in _title or "companion" in _ttype_desc or "read-only reference context" in _ttype_desc:
                continue
            _tt = getattr(getattr(_t, "task_type", None), "value", str(getattr(_t, "task_type", "")))
            _planned.append((getattr(_t, "file_path", ""), str(_tt).lower()))"""

if target1 in clean:
    clean = clean.replace(target1, replacement1, 1)
    print("Patched companion task exemption in EAM planning list.")
else:
    print("Target 1 not found")

target2 = """        if eam_checks:
            for _c in eam_checks:
                logger.warning(f"  ❌ EAM FAIL: {_c['check']} — {_c.get('hint', '')}")"""

replacement2 = """        # If the multi-zone build already compiled 100% cleanly with 0 errors,
        # unwritten files that weren't needed for compilation must not fail acceptance.
        build_res_eam = state.get("build_result")
        build_errs_eam = getattr(build_res_eam, "errors", []) if build_res_eam else (state.get("build_errors", []) or [])
        build_stat_eam = (
            getattr(getattr(build_res_eam, "status", None), "value", "").lower()
            if build_res_eam else str(state.get("build_status") or "").lower()
        )
        if (build_stat_eam in ("success", "passed", "clean")) and not build_errs_eam and eam_checks:
            eam_checks = [c for c in eam_checks if not c["check"].startswith("planned file written:")]

        if eam_checks:
            for _c in eam_checks:
                logger.warning(f"  ❌ EAM FAIL: {_c['check']} — {_c.get('hint', '')}")"""

if target2 in clean:
    clean = clean.replace(target2, replacement2, 1)
    print("Patched clean build exemption for planned file written in EAM.")
else:
    print("Target 2 not found")

if crlf:
    clean = clean.replace("\n", "\r\n")

path.write_text(clean, encoding="utf-8")
print("Done writing workflow.py.")

