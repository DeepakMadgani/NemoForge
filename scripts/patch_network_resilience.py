import re
from pathlib import Path

# 1. Patch llm_utils.py to retry DNS blips up to 3 times
lu_path = Path("nemoforge/src/ticket_to_code/llm_utils.py")
lu_content = lu_path.read_text(encoding="utf-8")

target_lu = '''            if is_fatal:
                if attempt == 0 and max_retries > 1:
                    logger.warning(f"LLM DNS hiccup on attempt 1: {exc}. Retrying once in 3s...")
                    time.sleep(3.0)
                    last_exc = exc
                    continue
                logger.error(f"LLM endpoint unreachable (DNS failure) — aborting without retry: {exc}")
                raise'''

repl_lu = '''            if is_fatal:
                if attempt < 3 and max_retries > 1:
                    wait_s = 2.0 * (attempt + 1)
                    logger.warning(f"LLM DNS hiccup on attempt {attempt + 1}: {exc}. Retrying in {wait_s}s...")
                    time.sleep(wait_s)
                    last_exc = exc
                    continue
                logger.error(f"LLM endpoint unreachable (DNS failure) — aborting after retries: {exc}")
                raise'''

if target_lu in lu_content:
    lu_content = lu_content.replace(target_lu, repl_lu, 1)
    lu_path.write_text(lu_content, encoding="utf-8")
    print("Successfully patched llm_utils.py DNS retry")
else:
    print("target_lu not found in llm_utils.py")

# 2. Patch code_generator.py to retry transient errors
cg_path = Path("nemoforge/src/ticket_to_code/agents/code_generator.py")
cg_content = cg_path.read_text(encoding="utf-8")

target_cg = '''            except Exception as e:
                logger.error(f"Code generation failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to generate code for task {task.id}: {e}")'''

repl_cg = '''            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Generation encountered transient error (attempt {attempt + 1}/{max_retries}), retrying in 2s: {e}")
                    import time as _t; _t.sleep(2.0)
                    continue
                logger.error(f"Code generation failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to generate code for task {task.id}: {e}")'''

if target_cg in cg_content:
    cg_content = cg_content.replace(target_cg, repl_cg, 1)
    cg_path.write_text(cg_content, encoding="utf-8")
    print("Successfully patched code_generator.py transient error retry")
else:
    print("target_cg not found in code_generator.py")

