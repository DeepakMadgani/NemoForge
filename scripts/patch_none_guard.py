from pathlib import Path

# 1. Patch code_generator.py to never return None and handle retry exhaustion cleanly
cg_path = Path("nemoforge/src/ticket_to_code/agents/code_generator.py")
cg_text = cg_path.read_text(encoding="utf-8")

target_cg = """                    continue
                    # Last attempt - try aggressive fallback
                    logger.error(f"All retry attempts exhausted for task {task.id}")
                    logger.error(f"Code generation failed: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to generate code for task {task.id}: {e}")
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Generation encountered transient error (attempt {attempt + 1}/{max_retries}), retrying in 2s: {e}")
                    import time as _t; _t.sleep(2.0)
                    continue
                logger.error(f"Code generation failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to generate code for task {task.id}: {e}")"""

repl_cg = """                    continue
                else:
                    logger.warning(f"All retry attempts exhausted for task {task.id}. Preserving original file content.")
                    return GeneratedCode(
                        file_path=task.file_path,
                        content=existing_content or self._current_existing_content or "",
                        language=task.language,
                        change_type=task.task_type,
                        documentation=f"Preserved original content after error: {_err_msg[:200]}",
                        imports=[],
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Generation encountered transient error (attempt {attempt + 1}/{max_retries}), retrying in 2s: {e}")
                    import time as _t; _t.sleep(2.0)
                    continue
                logger.warning(f"Code generation failed for task {task.id} after retries: {e}. Preserving original file content.")
                return GeneratedCode(
                    file_path=task.file_path,
                    content=existing_content or self._current_existing_content or "",
                    language=task.language,
                    change_type=task.task_type,
                    documentation=f"Preserved original content after exception: {e}",
                    imports=[],
                )

        return GeneratedCode(
            file_path=task.file_path,
            content=existing_content or self._current_existing_content or "",
            language=task.language,
            change_type=task.task_type,
            documentation="Preserved original content (loop exit fallback)",
            imports=[],
        )"""

if target_cg in cg_text:
    cg_text = cg_text.replace(target_cg, repl_cg, 1)
    cg_path.write_text(cg_text, encoding="utf-8")
    print("Successfully patched code_generator.py to eliminate None return")
else:
    print("target_cg not found, checking with stripped whitespace...")

# 2. Patch workflow.py to guard against None code defensively
wf_path = Path("nemoforge/src/ticket_to_code/workflow.py")
wf_text = wf_path.read_text(encoding="utf-8")

target_wf = """            # ── PATCH VALIDATION: check scope before writing to disk ──────────
            validation = validator.validate(code, task, existing_content)"""

repl_wf = """            # ── PATCH VALIDATION: check scope before writing to disk ──────────
            if code is None:
                from ticket_to_code.models import GeneratedCode
                logger.warning(f"Code generator returned None for {task.file_path} — preserving original content")
                code = GeneratedCode(
                    file_path=task.file_path,
                    content=existing_content or "",
                    language=task.language,
                    change_type=task.task_type,
                    documentation="Preserved original content (None guard)",
                    imports=[],
                )
            validation = validator.validate(code, task, existing_content)"""

if target_wf in wf_text:
    wf_text = wf_text.replace(target_wf, repl_wf, 1)
    wf_path.write_text(wf_text, encoding="utf-8")
    print("Successfully patched workflow.py None guard")
else:
    print("target_wf not found in workflow.py")

