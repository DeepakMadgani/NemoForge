"""
Runtime utilities: RunContext, RunBudget, DegradationRegistry.
"""

from ticket_to_code.runtime.run_context import (
    HealthLevel,
    RunBudget,
    BudgetExceeded,
    DegradationRegistry,
    RunContext,
    RunRecord,
    TokenCallRecord,
    PhaseUsage,
)
from ticket_to_code.runtime.intake_contract import (
    TicketKind,
    Priority,
    RiskLevel,
    RunStatus,
    ValidationResult,
    GroundTruthTicket,
    SuccessPolicy,
    ConstraintPolicy,
    StageRetryPolicy,
    DecisionPriority,
    DecisionRecord,
    ChangedFileRecord,
    ValidationEvidenceRecord,
    TicketRunReport,
    OrchestratorPolicy,
    enforce_hard_gates,
    summarize_validation,
    report_to_dict,
)
from ticket_to_code.runtime.orchestrator_skeleton import (
    StageFns,
    ContractOrchestrator,
)
from ticket_to_code.runtime.stage_depth import (
    StageDepthInput,
    StageDepthDecision,
    choose_stage_depth,
    apply_gate_overrides,
)
from ticket_to_code.runtime.agent_harness import (
    AgentIntent,
    classify_agent_intent,
    classify_agent_intent_hybrid,
    HostStateMetadata,
    UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS,
    PromptAssembler,
    ContextCompactor,
    HostEnvironmentTools,
    apply_tolerant_replacement,
    AgenticHarness,
)
from ticket_to_code.runtime.deductive_blackboard import (
    DeductiveBlackboard,
    ObservationRecord,
)
from ticket_to_code.runtime.tool_registry import (
    HighDensitySanitizer,
    UnifiedToolRegistry,
)
from ticket_to_code.runtime.multi_service_orchestrator import (
    ArchitectureLayer,
    LayerRequirement,
    MultiServiceOrchestrator,
)
from ticket_to_code.runtime.compiler_gate import (
    CompilerGate,
    CompilationDiagnostic,
    CompilationResult,
)
from ticket_to_code.runtime.polyglot_build import (
    PolyglotBuildDiscoverer,
    BuildConfig,
)
from ticket_to_code.runtime.git_sandbox import (
    GitWorktreeSandbox,
    SandboxEnvironment,
)
from ticket_to_code.runtime.blast_radius import (
    BlastRadiusAnalyzer,
    BlastRadiusReport,
)
from ticket_to_code.runtime.context_mentions import (
    ContextMentionResolver,
    MentionExpansionResult,
    ResolvedMention,
)
from ticket_to_code.runtime.interactive_gate import (
    InteractiveClarificationGate,
    ClarificationOption,
    ClarificationRequest,
    ClarificationResponse,
)
from ticket_to_code.runtime.diff_engine import (
    DiffReviewEngine,
    DiffHunk,
    FileDiff,
)
from ticket_to_code.runtime.telemetry_evals import (
    RuntimeTelemetryTracker,
    ToolTelemetry,
)

__all__ = [
    "ContextMentionResolver",
    "MentionExpansionResult",
    "ResolvedMention",
    "InteractiveClarificationGate",
    "ClarificationOption",
    "ClarificationRequest",
    "ClarificationResponse",
    "DiffReviewEngine",
    "DiffHunk",
    "FileDiff",
    "RuntimeTelemetryTracker",
    "ToolTelemetry",
    "HealthLevel",
    "RunBudget",
    "BudgetExceeded",
    "DegradationRegistry",
    "RunContext",
    "RunRecord",
    "TicketKind",
    "Priority",
    "RiskLevel",
    "RunStatus",
    "ValidationResult",
    "GroundTruthTicket",
    "SuccessPolicy",
    "ConstraintPolicy",
    "StageRetryPolicy",
    "DecisionPriority",
    "DecisionRecord",
    "ChangedFileRecord",
    "ValidationEvidenceRecord",
    "TicketRunReport",
    "OrchestratorPolicy",
    "enforce_hard_gates",
    "summarize_validation",
    "report_to_dict",
    "StageFns",
    "ContractOrchestrator",
    "StageDepthInput",
    "StageDepthDecision",
    "choose_stage_depth",
    "apply_gate_overrides",
    "AgentIntent",
    "classify_agent_intent",
    "classify_agent_intent_hybrid",
    "HostStateMetadata",
    "UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS",
    "PromptAssembler",
    "ContextCompactor",
    "HostEnvironmentTools",
    "apply_tolerant_replacement",
    "AgenticHarness",
    "DeductiveBlackboard",
    "ObservationRecord",
    "HighDensitySanitizer",
    "UnifiedToolRegistry",
    "ArchitectureLayer",
    "LayerRequirement",
    "MultiServiceOrchestrator",
    "CompilerGate",
    "CompilationDiagnostic",
    "CompilationResult",
    "PolyglotBuildDiscoverer",
    "BuildConfig",
    "GitWorktreeSandbox",
    "SandboxEnvironment",
    "BlastRadiusAnalyzer",
    "BlastRadiusReport",
]

