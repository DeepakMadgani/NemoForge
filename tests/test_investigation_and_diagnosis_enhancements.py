"""
Tests for Investigation, Diagnosis, Repo-Tree Reasoning, and Requirement Invariants.
"""
import re
import pytest
from pathlib import Path
from ticket_to_code.models import (
    ValueEdgeTicket,
    TicketType,
    InvestigationResult,
    DevelopmentTask,
    TaskType,
)
from ticket_to_code.agents.runtime_log_analyzer import (
    RuntimeLogAnalyzer,
    RuntimeDiagnosis,
)


def test_extract_error_code_and_entities():
    analyzer = RuntimeLogAnalyzer(workspace_path=".")
    
    # 1. Error code extraction
    log_sample = """
    2026-10-28 14:02:11 ERROR [OrderProcessingService] Failed to process order ORD-9921
    Status code: 409 Conflict. ErrorCode: ERR_SETTLEMENT_HOLD_FAILED
    Exception in thread "main" java.lang.IllegalStateException: PaymentHold is still active
    """
    code = analyzer.extract_error_code(log_sample)
    assert code in ("ERR_SETTLEMENT_HOLD_FAILED", "409", "409 Conflict")

    # TypeScript error code
    ts_log = "src/components/RefundModal.tsx:42:15 - error TS2339: Property 'refundAmount' does not exist on type 'Order'."
    assert analyzer.extract_error_code(ts_log) == "TS2339"

    # 2. Entity extraction
    entities = analyzer.extract_affected_entities(
        error_message="PaymentHold is still active on SettlementAccount",
        ticket_text="Refund order CF-REFUND-101 and release PaymentHold in settlement-service",
    )
    assert "PaymentHold" in entities
    assert any(e in entities for e in ["Order", "SettlementAccount", "Settlement"])


def test_determine_failure_origin_python_vs_java():
    analyzer = RuntimeLogAnalyzer(workspace_path=".")
    
    # Python traceback: stack is bottom-up (root failure is last frame in repo)
    py_frames = [
        {"file": "C:/Python313/Lib/site-packages/uvicorn/main.py", "line": 402, "is_repo": False},
        {"file": "./src/ticket_to_code/workflow.py", "line": 150, "is_repo": True},
        {"file": "./src/ticket_to_code/agents/runtime_log_analyzer.py", "line": 95, "is_repo": True},
    ]
    py_file, py_line = analyzer._determine_failure_origin(py_frames, is_python=True)
    assert py_file.endswith("runtime_log_analyzer.py")
    assert py_line == 95

    # Java traceback: stack is top-down (root failure is top repo frame, ignoring java stdlib/framework)
    java_frames = [
        {"file": "java/lang/Thread.java", "line": 833, "is_repo": False},
        {"file": "com/cloudforge/commerce/order/service/OrderService.java", "line": 142, "is_repo": True},
        {"file": "com/cloudforge/commerce/order/controller/OrderController.java", "line": 88, "is_repo": True},
    ]
    java_file, java_line = analyzer._determine_failure_origin(java_frames, is_python=False)
    assert java_file.endswith("OrderService.java")
    assert java_line == 142


def test_development_task_requirement_ids():
    task = DevelopmentTask(
        id="task-1",
        title="Add REFUNDED status to Order",
        file_path="order-service/src/main/java/com/cloudforge/order/model/OrderStatus.java",
        task_type=TaskType.MODIFY,
        requirement_ids=["REQ-1", "REQ-2"],
        description="Extend enum with REFUNDED",
    )
    assert task.requirement_ids == ["REQ-1", "REQ-2"]


def test_windows_path_normalization_and_jvm_build_tool():
    from ticket_to_code.agents.diagnostic_normalizer import _normalize_diagnostic_path, _try_parse

    raw_path = "/C:/Users/test/Order.java"
    normalized = _normalize_diagnostic_path(raw_path)
    assert normalized == "C:/Users/test/Order.java"

    # Maven line parsing
    maven_line = "[ERROR] /C:/Users/test/OrderController.java:[45,38] cannot find symbol"
    diag = _try_parse(maven_line)
    assert diag is not None
    assert diag.source_file == "C:/Users/test/OrderController.java"


def test_localization_agent_preserves_new_planner_methods(tmp_path):
    from ticket_to_code.agents.localization_agent import LocalizationAgent
    from ticket_to_code.models import DevelopmentTask, TaskType

    test_file = tmp_path / "OrderService.java"
    test_file.write_text("""
public class OrderService {
    public Order placeOrder(OrderReq req) { return null; }
    public Order cancelOrder(Long id) { return null; }
}
""", encoding="utf-8")

    agent = LocalizationAgent(workspace_path=str(tmp_path))
    task = DevelopmentTask(
        id="task-test",
        title="Implement refund transaction",
        file_path="OrderService.java",
        task_type=TaskType.MODIFY,
        allowed_methods=["refundOrder"],
        description="Add refundOrder method to handle refund flow and release ledger hold",
    )
    agent._resolve_method_targets(task, TaskType)
    assert "refundOrder" in task.allowed_methods
    assert task.target_method == "refundOrder"


def test_numbered_requirements_extraction_order():
    desc = """Allow Store Operators to Refund Fulfilled Orders:

Store operations require the capability to issue refunds on fulfilled orders.

Requirements:
1. Extend the order lifecycle status domain to support a REFUNDED state.
2. Implement the refund transaction in OrderService.
3. Expose a REST endpoint allowing operators to submit refund requests.
4. In the web management UI, add a Refund action for orders in FULFILLED status.
5. Ensure all builds compile cleanly.
"""
    numbered_items = []
    for d_line in desc.splitlines():
        nm = re.match(r"^\s*(\d+)[\.\)]\s+(.+)$", d_line.strip())
        if nm:
            numbered_items.append((int(nm.group(1)), nm.group(2).strip()))
    
    numbered_items.sort(key=lambda x: x[0])
    raw_func_reqs = [text for _, text in numbered_items]
    
    formatted = [f"[REQ-{idx}] {r}" for idx, r in enumerate(raw_func_reqs, start=1)]
    assert len(formatted) == 5
    assert formatted[0].startswith("[REQ-1] Extend the order lifecycle")
    assert formatted[1].startswith("[REQ-2] Implement the refund transaction")
    assert formatted[2].startswith("[REQ-3] Expose a REST endpoint")
    assert formatted[3].startswith("[REQ-4] In the web management UI")
    assert formatted[4].startswith("[REQ-5] Ensure all builds compile cleanly")
