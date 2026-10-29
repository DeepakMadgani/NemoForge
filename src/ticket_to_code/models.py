"""
Data Models for Autonomous Ticket-to-Code System

All Pydantic models for structured data validation and AI output parsing.

Author: Deepak Madgani
Date: April 2026
"""

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
import time
from dataclasses import dataclass, asdict

# ============================================================================
# TICKET MODELS
# ============================================================================

class TicketPriority(str, Enum):
    """Ticket priority levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TicketType(str, Enum):
    """Type of ticket/issue"""
    FEATURE_REQUEST = "feature_request"           # New functionality
    BUG_FIX = "bug_fix"                          # Code defect needing fix
    INVESTIGATION = "investigation"               # Problem analysis needed
    CONFIGURATION = "configuration"               # Config/setup issue
    DOCUMENTATION = "documentation"               # Docs only
    REFACTORING = "refactoring"                  # Code improvement
    PERFORMANCE = "performance"                   # Performance issue
    DEPLOYMENT = "deployment"                     # Deployment-related
    USAGE_QUESTION = "usage_question"            # How-to question
    VERSION_BUMP = "version_bump"                # Version string update
    API_CHANGE = "api_change"                    # API/endpoint change
    DATABASE = "database"                        # Database/SQL change
    SECURITY = "security"                        # Security-related fix
    DEPENDENCY = "dependency"                    # Dependency upgrade
    UI = "ui"                                    # Pure UI/CSS/layout change


# ============================================================================
# TECHNICAL FACT EXTRACTOR MODELS
# These are populated deterministically (zero LLM calls) before investigation.
# They guarantee that the LLM cannot ignore critical technical signals.
# ============================================================================

class FileFact(BaseModel):
    """A file name or path explicitly mentioned in the ticket."""
    name: str = Field(..., description="Filename (e.g., 'run-job.sh')")
    path: Optional[str] = Field(None, description="Full or partial path if mentioned")
    operation: str = Field(
        default="modify",
        description="Inferred operation: 'modify' | 'create' | 'delete' | 'unknown'"
    )


class VersionFact(BaseModel):
    """Represents a version transition extracted from ticket text."""
    old_value: str = Field(..., description="Current/old version value")
    new_value: str = Field(..., description="Target/new version value")
    permutations_old: List[str] = Field(
        default_factory=list,
        description="Normalized search variants for old version"
    )
    permutations_new: List[str] = Field(
        default_factory=list,
        description="Normalized replace variants for new version"
    )


class TechnicalFacts(BaseModel):
    """
    All deterministically extracted technical facts from a ticket.

    Populated by TicketFactExtractor (pure Python, zero LLM calls) before
    the LLM investigation step, so critical details like version numbers and
    explicit file names cannot be missed by the LLM.
    """
    versions: List[VersionFact] = Field(default_factory=list, description="Version transitions detected in ticket")
    files: List[FileFact] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list, description="PascalCase class/component names found in ticket")
    methods: List[str] = Field(default_factory=list, description="camelCase method names found in ticket")
    config_keys: List[str] = Field(default_factory=list, description="Config keys like 'application.version'")
    paths: List[str] = Field(default_factory=list, description="Directory paths like 'src/app'")
    error_codes: List[str] = Field(default_factory=list, description="Error codes like 'IDX10000', 'NullReferenceException'")
    urls: List[str] = Field(default_factory=list, description="URLs found in ticket")
    sql_snippets: List[str] = Field(default_factory=list, description="SQL patterns detected")
    api_routes: List[str] = Field(default_factory=list, description="API Routes (e.g. /api/v1/users) found in ticket")
    classified_type: str = Field(
        default="unknown",
        description="Deterministic ticket type: VERSION_BUMP | UI | BUG | CONFIG | API | DATABASE | SECURITY | DEPENDENCY | FEATURE"
    )
    section_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Ticket split by section: {'description': '...', 'technical_context': '...', 'files': '...'}"
    )
    high_priority_literals: List[str] = Field(
        default_factory=list,
        description="Flattened list of all high-priority search literals ready for query expansion"
    )
    goals: List["GoalFact"] = Field(
        default_factory=list,
        description="High-level business intent of this hypothesis (e.g., 'version bump', 'css fix')."
    )

    def has_facts(self) -> bool:
        """Return True if any concrete technical facts were extracted."""
        return bool(
            self.files or self.classes or
            self.methods or self.config_keys or self.error_codes
        )


class ClassifiedType(BaseModel):
    """
    Result of the deterministic ticket classifier.
    Returns primary + optional secondary type with confidence.
    A ticket can belong to multiple categories (e.g., VERSION_BUMP + UI).
    """
    primary: str = Field(..., description="Primary ticket type: VERSION_BUMP | UI | BUG | CONFIG | API | DATABASE | SECURITY | DEPENDENCY | FEATURE")
    secondary: Optional[str] = Field(None, description="Optional secondary type when ticket spans two categories")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    signals: List[str] = Field(default_factory=list, description="Human-readable reasons for this classification")

    def penalty_map(self) -> Dict[str, float]:
        """
        Returns a file extension → score multiplier map.
        Extensions in this map get their evidence score multiplied down.
        NOT a hard blacklist — a file can still be selected if evidence is strong.
        """
        maps = {
            "VERSION_BUMP": {".scss": 0.3, ".html": 0.3, ".spec.ts": 0.1, ".test.ts": 0.1},
            "UI":           {".sh": 0.2, ".properties": 0.2, ".java": 0.3, ".py": 0.3},
            "CONFIG":       {".ts": 0.3, ".java": 0.3, ".cs": 0.3, ".py": 0.3},
            "DATABASE":     {".ts": 0.2, ".html": 0.1, ".scss": 0.1},
            "SECURITY":     {},   # any file could be relevant
            "BUG":          {},   # any file could be root cause
            "DEPENDENCY":   {".ts": 0.2, ".html": 0.1},
            "API":          {".scss": 0.1, ".html": 0.2},
            "FEATURE":      {},
        }
        base = maps.get(self.primary, {})
        # Merge secondary penalties at half weight
        if self.secondary and self.secondary in maps:
            for ext, penalty in maps[self.secondary].items():
                base[ext] = min(base.get(ext, 1.0), penalty * 0.5 + 0.5)
        return base


class SymbolFact(BaseModel):
    """A code symbol (class, method, constant) explicitly mentioned in the ticket."""
    name: str
    kind: str = Field(default="unknown", description="class | method | constant | component | interface")
    expansion: List[str] = Field(default_factory=list, description="Expanded search variants")


class ConstraintFact(BaseModel):
    """An explicit constraint mentioned in the ticket or conversation."""
    description: str
    scope: str = Field(default="unknown", description="file | module | layer | technology")
    constraint_type: str = Field(default="restriction", description="restriction | requirement | preference")


class GoalFact(BaseModel):
    """The user's high-level goal extracted from the ticket."""
    description: str
    acceptance_criteria: List[str] = Field(default_factory=list)


class PlannerContext(BaseModel):
    """
    Full context given to the Planner node.
    The planner should NEVER need to retrieve anything — everything is here.
    """
    classified_type: Optional[ClassifiedType] = None
    technical_facts: Optional[TechnicalFacts] = None
    allowed_files: List[str] = Field(default_factory=list, description="Files ranked by evidence engine")
    allowed_symbols: List[str] = Field(default_factory=list, description="Symbols from SymbolFact + evidence")
    acceptance_criteria: List[str] = Field(default_factory=list)
    constraints: List[ConstraintFact] = Field(default_factory=list)
    penalty_map: Dict[str, float] = Field(default_factory=dict, description="file_ext → score multiplier")
    workspace_summary: Optional[str] = Field(None, description="Brief workspace architecture summary")
    context_health: Optional["ContextHealth"] = Field(None, description="Metrics on context retrieval health")


class ExecutionState(BaseModel):
    """
    Stage-level retry counters — kept separate from TicketToCodeState.
    Each stage has its own independent retry budget.
    """
    plan_retry: int = 0
    retrieval_retry: int = 0
    generation_retry: int = 0
    build_retry: int = 0
    max_plan_retries: int = 3
    max_retrieval_retries: int = 3
    max_generation_retries: int = 5
    max_build_retries: int = 3

    def can_retry_plan(self) -> bool:
        return self.plan_retry < self.max_plan_retries

    def can_retry_retrieval(self) -> bool:
        return self.retrieval_retry < self.max_retrieval_retries

    def can_retry_generation(self) -> bool:
        return self.generation_retry < self.max_generation_retries

    def can_retry_build(self) -> bool:
        return self.build_retry < self.max_build_retries



class ValueEdgeTicket(BaseModel):
    """ValueEdge ticket input model"""
    ticket_id: str = Field(..., description="Ticket ID (e.g., VE-12345)")
    title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Detailed description")
    acceptance_criteria: List[str] = Field(
        default_factory=list, 
        description="List of acceptance criteria"
    )
    labels: List[str] = Field(default_factory=list, description="Ticket labels/tags")
    priority: TicketPriority = Field(
        default=TicketPriority.MEDIUM, 
        description="Ticket priority"
    )
    assignee: Optional[str] = Field(None, description="Assigned developer")
    created_date: Optional[datetime] = Field(None, description="Creation timestamp")
    attachments: List[str] = Field(
        default_factory=list,
        description="List of file paths or URLs for attached screenshots, logs, or configs"
    )
    # Ticket-declared change authorization scope (evidence is NOT authorization).
    expected_changed_files: List[str] = Field(
        default_factory=list,
        description="Files the ticket authorizes for modification"
    )
    expected_owner_files: List[str] = Field(
        default_factory=list,
        description="Primary owner files for the ticket's behavioral delta"
    )
    forbidden_files: List[str] = Field(
        default_factory=list,
        description="Files that must never be modified for this ticket"
    )
    
    def to_summary(self) -> str:
        """Generate human-readable summary"""
        return f"""
Ticket: {self.ticket_id}
Title: {self.title}
Priority: {self.priority.value}
Description: {self.description}
Acceptance Criteria:
{chr(10).join(f"  - {ac}" for ac in self.acceptance_criteria)}
        """.strip()


class InvestigationResult(BaseModel):
    """Result of ticket investigation/triage"""
    ticket_type: TicketType = Field(..., description="Classified ticket type")
    requires_code_changes: bool = Field(
        ..., 
        description="Whether code changes are needed"
    )
    root_cause_hypothesis: Optional[str] = Field(
        None,
        description="Suspected root cause (for bugs/issues)"
    )
    affected_systems: List[str] = Field(
        default_factory=list,
        description="Systems/services likely affected"
    )
    core_subjects: List[str] = Field(
        default_factory=list,
        description="Core entities/values being changed (e.g., 'version number', 'deliverable count')"
    )
    context_locations: List[str] = Field(
        default_factory=list,
        description="Contextual locations where the issue manifests (e.g., 'dashboard', 'CC4E app')"
    )
    investigation_areas: List[str] = Field(
        default_factory=list,
        description="Areas to investigate (config, code, data, etc.)"
    )
    possible_causes: List[str] = Field(
        default_factory=list,
        description="List of possible causes"
    )
    # New Observability Fields
    goal: Optional[str] = Field(None, description="Goal of the ticket")
    constraints: List[str] = Field(default_factory=list, description="Constraints")
    literal_values: List[str] = Field(default_factory=list, description="Literal values extracted")
    expected_outcome: Optional[str] = Field(None, description="Expected outcome")
    clarification_needed: bool = Field(False, description="Clarification needed?")
    explanation: str = Field(
        default="",
        description="Human-readable explanation of findings"
    )
    needs_more_context: bool = Field(
        default=False,
        description="Whether more context is needed before proceeding"
    )
    current_state_literals: List[str] = Field(
        default_factory=list,
        description="Exact values/strings that exist in the code TODAY (what to search for)."
    )
    desired_state_literals: List[str] = Field(
        default_factory=list,
        description="Exact values/strings the ticket wants the code to become (target state)."
    )

    recommended_action: str = Field(
        ...,
        description="What should be done next"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis"
    )
    
class StrategyDecision(BaseModel):
    """Strategy selected for this ticket category and execution policy."""
    category: str = Field(..., description="Primary strategy category")
    strategy_profile: str = Field(..., description="Execution profile name")
    required_evidence: List[str] = Field(default_factory=list)
    required_validations: List[str] = Field(default_factory=list)
    stop_conditions: List[str] = Field(default_factory=list)
    skill_tags: List[str] = Field(default_factory=list, description="Skill identifiers to load")
    rationale: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    explanation: str = Field(
        ...,
        description="Human-readable explanation of findings"
    )
    needs_more_context: bool = Field(
        default=False,
        description="Whether more context is needed before proceeding"
    )
    current_state_literals: List[str] = Field(
        default_factory=list,
        description="Exact values/strings that exist in the code TODAY (what to search for). These are the PRIMARY search targets."
    )
    desired_state_literals: List[str] = Field(
        default_factory=list,
        description="Exact values/strings the ticket wants the code to become (the target). NOT search terms — they don't exist in the codebase yet."
    )


# ============================================================================
# THINKING CHAIN — Inter-Stage Reasoning Memory
# Each pipeline stage records what it noticed and decided, so the next LLM
# call receives that compacted reasoning as explicit context instead of having
# to re-derive it from scratch.  This is the "agent-to-agent thought handoff".
# ============================================================================

class StageThought(BaseModel):
    """Structured reasoning record produced by one pipeline stage."""

    stage: str = Field(..., description="Stage name: investigation | planning | localization | codegen")
    summary: str = Field(..., description="One-paragraph synthesis of what this stage concluded")
    key_decisions: List[str] = Field(
        default_factory=list,
        description="Bullet-point decisions made (files chosen, hypotheses confirmed, etc.)"
    )
    signals_noted: List[str] = Field(
        default_factory=list,
        description="Evidence signals that influenced this stage's decisions"
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    raw_thinking: Optional[str] = Field(
        None,
        description="Raw <thinking>...</thinking> block captured from the LLM response, if any"
    )

    def to_prompt_block(self) -> str:
        """Render as a compact block for injection into the next stage's prompt."""
        lines = [f"## [{self.stage.upper()} REASONING] (confidence={self.confidence:.2f})"]
        lines.append(self.summary)
        if self.signals_noted:
            lines.append("Signals noticed: " + "; ".join(self.signals_noted[:5]))
        if self.key_decisions:
            lines.append("Key decisions:")
            for d in self.key_decisions[:6]:
                lines.append(f"  • {d}")
        return "\n".join(lines)


class ThinkingChain(BaseModel):
    """Accumulated reasoning from all pipeline stages so far."""

    thoughts: List[StageThought] = Field(default_factory=list)

    def append(self, thought: StageThought) -> "ThinkingChain":
        """Return a new chain with the thought appended (immutable-friendly)."""
        return ThinkingChain(thoughts=self.thoughts + [thought])

    def to_prompt_block(self) -> str:
        """Render all accumulated thoughts as a single context block for the next stage."""
        if not self.thoughts:
            return ""
        lines = [
            "=" * 70,
            "ACCUMULATED REASONING FROM PREVIOUS PIPELINE STAGES",
            "(READ THIS CAREFULLY — build on it, do not re-derive what's already known)",
            "=" * 70,
        ]
        for t in self.thoughts:
            lines.append("")
            lines.append(t.to_prompt_block())
        lines.append("=" * 70)
        return "\n".join(lines)

    def last(self) -> Optional[StageThought]:
        return self.thoughts[-1] if self.thoughts else None


# ============================================================================


class StructuredRequirements(BaseModel):
    """Analyzed and structured requirements from ticket"""
    functional_requirements: List[str] = Field(
        ..., 
        description="What the system should do"
    )
    technical_requirements: List[str] = Field(
        ..., 
        description="How to implement (technical details)"
    )
    edge_cases: List[str] = Field(
        ..., 
        description="Edge cases and error scenarios to handle"
    )
    acceptance_tests: List[str] = Field(
        ..., 
        description="Test cases for acceptance criteria"
    )
    affected_components: List[str] = Field(
        ..., 
        description="Components/modules that will be modified"
    )
    non_functional_requirements: List[str] = Field(
        default_factory=list,
        description="Performance, security, scalability requirements"
    )


class SolutionGuidance(BaseModel):
    """
    Solution guidance for issues that don't require code changes.
    
    For configuration issues, usage questions, deployment problems, etc.
    """
    issue_summary: str = Field(
        ...,
        description="Clear summary of the issue"
    )
    root_cause: str = Field(
        ...,
        description="Identified root cause"
    )
    solution_type: str = Field(
        ...,
        description="Type of solution (configuration, usage, deployment, etc.)"
    )
    step_by_step_solution: List[str] = Field(
        ...,
        description="Step-by-step instructions to resolve the issue"
    )
    configuration_examples: List[str] = Field(
        default_factory=list,
        description="Configuration file examples or code snippets"
    )
    verification_steps: List[str] = Field(
        ...,
        description="How to verify the issue is resolved"
    )

class VerificationStatus(str, Enum):
    VERIFIED_EXISTS = "VERIFIED_EXISTS"
    VERIFIED_ABSENT = "VERIFIED_ABSENT"
    UNKNOWN = "UNKNOWN"

class VerificationEvidence(BaseModel):
    file: str
    symbol: Optional[str] = None
    reason: str

class PlanningScopeVerificationResult(BaseModel):
    claim: str
    status: VerificationStatus
    evidence: List[VerificationEvidence] = Field(default_factory=list)
    planning_implication: str
    common_mistakes: List[str] = Field(
        default_factory=list,
        description="Common mistakes that cause this issue"
    )
    related_documentation: List[str] = Field(
        default_factory=list,
        description="Links or references to relevant documentation"
    )
    escalation_criteria: Optional[str] = Field(
        None,
        description="When to escalate (if solution doesn't work)"
    )


# ============================================================================
# REPOSITORY KNOWLEDGE BASE (RKB) MODELS
# ============================================================================

class ArtifactRole(str, Enum):
    """Role classification for an artifact within a capability"""
    PRIMARY_LOGIC = "PRIMARY_LOGIC"
    SUPPORTING_CONTEXT = "SUPPORTING_CONTEXT"
    SHARED_COMPONENT = "SHARED_COMPONENT"
    TRANSLATION = "TRANSLATION"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    CONFIGURATION = "CONFIGURATION"


class CapabilityNode(BaseModel):
    """A specific capability executed within a workflow"""
    name: str = Field(..., description="Capability name")
    purpose: str = Field(..., description="Business purpose of this capability")
    user_actions: List[str] = Field(..., description="User actions that trigger this capability")
    primary_artifacts: List[str] = Field(..., description="Files serving as primary logic")
    supporting_artifacts: List[str] = Field(..., description="Files serving as supporting context")
    workflow_execution_path: str = Field(..., description="End-to-end execution path")


class WorkflowNode(BaseModel):
    """A business workflow comprising multiple capabilities"""
    name: str = Field(..., description="Workflow name")
    capabilities: List[CapabilityNode] = Field(..., description="Capabilities comprising this workflow")


class DomainNode(BaseModel):
    """A top-level business domain representing a major product module"""
    name: str = Field(..., description="Domain name")
    business_purpose: str = Field(..., description="Primary business purpose")
    workflows: List[WorkflowNode] = Field(..., description="Workflows within this domain")
    external_dependencies: List[str] = Field(..., description="Dependencies on other domains/services")
    confidence_score: float = Field(..., description="Evidence-backed confidence score")


class SemanticRetrievalResult(BaseModel):
    """Output from semantic, brain-based localization"""
    detected_domain: str = Field(..., description="Domain resolving the ticket")
    detected_workflow: str = Field(..., description="Workflow resolving the ticket")
    detected_capability: str = Field(..., description="Specific capability to modify")
    responsible_artifacts: List[str] = Field(..., description="Artifacts directly responsible")
    confidence: float = Field(..., description="Confidence in this semantic mapping")


# ============================================================================
# SEARCH CONTEXT MODELS (V3 Iterative Discovery)
# ============================================================================

class SearchAnchor(BaseModel):
    """An anchor extracted from code reading (e.g., a variable, service, or import)."""
    name: str = Field(..., description="The name of the anchor")
    kind: str = Field(..., description="The kind: e.g., 'variable', 'import', 'injectable', 'selector', 'method'")
    source_file: str = Field(..., description="Where it was extracted from")
    score: int = Field(default=0, description="Priority score given by AnchorPrioritizer")

class SearchBudget(BaseModel):
    """Budget for the V3 Iterative Discovery Engine."""
    start_time: float = Field(default_factory=time.time)
    max_seconds: int = 20
    searches_performed: int = 0
    max_searches: int = 15
    files_opened: int = 0
    max_files: int = 100
    graph_hops: int = 0
    max_hops: int = 100

    def is_exhausted(self) -> bool:
        if (time.time() - self.start_time) > self.max_seconds: return True
        if self.searches_performed >= self.max_searches: return True
        if self.files_opened >= self.max_files: return True
        if self.graph_hops >= self.max_hops: return True
        return False

class SearchContext(BaseModel):
    """
    Central state object for the V3 Iterative Discovery Engine.
    Tracks visited nodes to prevent cycles and stores the active search history.
    """
    ticket_facts: Optional[Any] = Field(None, description="TechnicalFacts passed in")
    
    visited_files: Set[str] = Field(default_factory=set)
    visited_symbols: Set[str] = Field(default_factory=set)
    visited_queries: Set[str] = Field(default_factory=set)
    visited_services: Set[str] = Field(default_factory=set)
    
    candidate_history: List[str] = Field(default_factory=list)
    anchor_history: List[SearchAnchor] = Field(default_factory=list)
    rejected_candidates: Set[str] = Field(default_factory=set)
    
    budget: SearchBudget = Field(default_factory=SearchBudget)
    iteration: int = 0
    has_escalated: bool = False


# ============================================================================
# BEHAVIOR INVESTIGATION MODELS
# ============================================================================

class BehaviorOwnerCategory(str, Enum):
    """Categorization of a file within a behavior graph"""
    VERIFIED_OWNER = "VERIFIED_OWNER"
    SUPPORTING_CONTEXT = "SUPPORTING_CONTEXT"
    REJECTED_CANDIDATE = "REJECTED_CANDIDATE"


class BehaviorNode(BaseModel):
    """A node representing a file in the Behavior Graph"""
    file_path: str = Field(..., description="Path to the file")
    role: str = Field(..., description="Role of the file in the behavior, e.g., 'renders count', 'applies filter'")
    relevant_methods: List[str] = Field(default_factory=list, description="Methods involved in the behavior flow")
    category: BehaviorOwnerCategory = Field(default=BehaviorOwnerCategory.REJECTED_CANDIDATE, description="Ownership category")
    snippet: Optional[str] = Field(None, description="Relevant code snippet justifying the node's role")


class BehaviorGraph(BaseModel):
    """The complete representation of the behavior execution flow"""
    ticket_id: Optional[str] = Field(None, description="Ticket ID")
    nodes: List[BehaviorNode] = Field(default_factory=list, description="Nodes (files) participating in the behavior")
    edges: List[str] = Field(default_factory=list, description="Descriptions of the data flow, e.g., 'A.foo() calls B.bar()'")
    suspected_bug: Optional[str] = Field(None, description="What is actually wrong in this behavior graph")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score of the graph accuracy")


class FactLocation(BaseModel):
    """Location of an extracted fact in the source code"""
    file: str = Field(..., description="File path")
    line: Optional[int] = Field(None, description="Line number")
    column: Optional[int] = Field(None, description="Column number")
    name: str = Field(..., description="The name of the fact (e.g. symbol name, method name)")

class FileFactExtract(BaseModel):
    """Raw technical facts extracted from a single file"""
    file_path: str = Field(..., description="Path to the file")
    artifact_type: str = Field(..., description="e.g. component, script, config")
    purpose: str = Field(..., description="Technical purpose of the file")
    symbols: List[FactLocation] = Field(default_factory=list, description="Important symbols (classes, interfaces)")
    methods: List[FactLocation] = Field(default_factory=list, description="Methods/functions defined")
    constants: List[FactLocation] = Field(default_factory=list, description="Constants defined or hardcoded strings")
    routes: List[FactLocation] = Field(default_factory=list, description="Routes handled or defined")
    services: List[str] = Field(default_factory=list, description="Services defined or instantiated")
    imports: List[str] = Field(default_factory=list, description="Significant imports")
    exports: List[str] = Field(default_factory=list, description="Significant exports")
    events: List[str] = Field(default_factory=list, description="Events emitted or listened to")
    state: List[str] = Field(default_factory=list, description="State managed or mutated")
    permissions: List[str] = Field(default_factory=list, description="Permissions or auth checked")
    config_keys: List[str] = Field(default_factory=list, description="Config keys used")
    external_apis: List[str] = Field(default_factory=list, description="External APIs called")

class FileFactReport(BaseModel):
    """The final output of the File Fact Extraction Agent"""
    ticket_id: str = Field(..., description="Ticket ID")
    file_facts: List[FileFactExtract] = Field(default_factory=list, description="Extracted facts for all candidates")

class CapabilityEvidence(BaseModel):
    """Evidence trace explaining why this capability exists"""
    symbols: List[FactLocation] = Field(default_factory=list, description="Symbols mapped to this capability")
    routes: List[FactLocation] = Field(default_factory=list, description="Routes mapped to this capability")
    constants: List[FactLocation] = Field(default_factory=list, description="Constants mapped to this capability")
    brain_summary: str = Field(default="", description="Summary from Brain memory")
    files: List[str] = Field(default_factory=list, description="Files providing this capability")

class FileParticipant(BaseModel):
    """A file that participates in a specific capability"""
    file_path: str = Field(..., description="Path to the file")
    role: str = Field(..., description="Role of the file in this capability")
    change_surfaces: List[str] = Field(default_factory=list, description="Symbols or exact code locations to change")
    artifact_type: str = Field(..., description="Artifact type of the file")

class CapabilityGraphNode(BaseModel):
    """A Business Capability representing the cross-file capability feature"""
    capability_id: str = Field(..., description="Unique semantic ID for this capability, e.g. version.application.store")
    business_domain: str = Field(..., description="Business domain")
    business_feature: str = Field(..., description="Business feature")
    capability_name: str = Field(..., description="Name of the capability")
    
    # Intent
    purpose: str = Field(..., description="Aggregated purpose of this capability")
    business_goal: str = Field(default="", description="Aggregated business goal")
    business_rules: List[str] = Field(default_factory=list, description="Aggregated business rules")
    
    # Mechanics
    triggers: List[str] = Field(default_factory=list, description="Events or actions triggering this")
    state_modified: List[str] = Field(default_factory=list, description="State modified by this capability")
    side_effects: List[str] = Field(default_factory=list, description="Known side effects")
    
    # Relationships
    depends_on: List[str] = Field(default_factory=list, description="Capability IDs this depends on")
    uses: List[str] = Field(default_factory=list, description="Capability IDs this uses")
    publishes: List[str] = Field(default_factory=list, description="Events/data published")
    consumes: List[str] = Field(default_factory=list, description="Events/data consumed")
    affected_by: List[str] = Field(default_factory=list, description="Capability IDs that affect this")
    
    business_concepts: List[str] = Field(default_factory=list, description="Business concepts associated")
    participating_files: List[FileParticipant] = Field(default_factory=list, description="Files participating in this capability")
    
    # Provenance and Tracing
    evidence: CapabilityEvidence = Field(default_factory=CapabilityEvidence, description="Trace explaining why this capability exists")
    confidence_sources: List[str] = Field(default_factory=list, description="Sources of confidence (e.g. ast, neo4j, brain, llm_summary)")
    version: int = Field(default=1, description="Version of this capability in the brain")

class ConsolidatedCapabilityReport(BaseModel):
    """Output from the LLM Capability Understanding Agent"""
    ticket_id: str = Field(..., description="Ticket ID")
    capabilities: List[CapabilityGraphNode] = Field(default_factory=list, description="Understood capabilities")

class CapabilityGraphReport(BaseModel):
    """The grouped semantic graph of capabilities"""
    ticket_id: str = Field(..., description="Ticket ID")
    capability_nodes: List[CapabilityGraphNode] = Field(default_factory=list, description="The graph of capabilities")

class RetrievedCapability(BaseModel):
    """A capability retrieved with its explanation"""
    capability: CapabilityGraphNode = Field(..., description="The capability retrieved")
    retrieval_reason: str = Field(..., description="Explanation of why this capability was retrieved for the ticket")

class RetrievedCapabilities(BaseModel):
    """The filtered subset of capabilities relevant to the ticket"""
    ticket_id: str = Field(..., description="Ticket ID")
    relevant_nodes: List[RetrievedCapability] = Field(default_factory=list, description="Relevant capabilities with reasons")


class AgentExecutionTrace(BaseModel):
    """Detailed execution trace for an agent stage"""
    stage_name: str = Field(..., description="Name of the pipeline stage")
    start_time: str = Field(..., description="ISO formatted start time")
    end_time: str = Field(..., description="ISO formatted end time")
    duration_ms: int = Field(..., description="Duration in milliseconds")
    
    # LLM Usage
    model: str = Field(default="", description="Model used")
    prompt_tokens: int = Field(default=0, description="Prompt tokens used")
    completion_tokens: int = Field(default=0, description="Completion tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")
    
    # Decisions
    accepted: List[Any] = Field(default_factory=list, description="Accepted items")
    rejected: List[Dict[str, Any]] = Field(default_factory=list, description="Rejected items with reasons")
    skipped: List[Dict[str, Any]] = Field(default_factory=list, description="Skipped items with reasons")
    
    # Custom stage metrics
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Stage specific metrics")
    decision_reasoning: Dict[str, Any] = Field(default_factory=dict, description="Explanations of why certain decisions were made")




class ValidationStatus(str, Enum):
    """Status of the plan validation"""
    APPROVED = "APPROVED"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"


class ValidationResult(BaseModel):
    """Output of the Plan Validation Stage"""
    status: ValidationStatus = Field(..., description="Validation outcome")
    failed_rules: List[str] = Field(default_factory=list, description="List of rule IDs that failed")
    hallucinated_files: List[str] = Field(default_factory=list, description="Files in the plan absent from retrieved capabilities")
    missing_evidence: List[str] = Field(default_factory=list, description="Tasks missing clear capability evidence or change surface")
    validated_plan: Optional['ValidatedPlan'] = Field(None, description="The constrained plan for generation if approved")


# ============================================================================
# LOCALIZATION MODELS
# ============================================================================

class TargetFile(BaseModel):
    """Target file identified for modification or creation"""
    file_path: str = Field(..., description="Relative file path")
    absolute_path: str = Field(..., description="Absolute file path")
    language: str = Field(..., description="Programming language")
    task_type: str = Field(..., description="create or modify")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(..., description="Why this file was selected")


class TargetMethod(BaseModel):
    """Target method within a file"""
    name: str = Field(..., description="Method name")
    fully_qualified_name: str = Field(..., description="Fully qualified name")
    file_path: str = Field(..., description="File containing this method")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    signature: Optional[str] = Field(None, description="Method signature")
    body: Optional[str] = Field(None, description="Full method body source code")
    operation: str = Field(..., description="modify, delete, or refactor")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ExecutionPath(BaseModel):
    """Execution path through the system"""
    start_component: str = Field(..., description="Starting component (e.g., Controller)")
    end_component: str = Field(..., description="Ending component (e.g., Repository)")
    intermediate_components: List[str] = Field(
        default_factory=list,
        description="Components in between"
    )
    path_length: int = Field(..., description="Number of hops in path")


class ImpactAnalysis(BaseModel):
    """Impact analysis of changes"""
    affected_apis: List[str] = Field(default_factory=list, description="Affected API endpoints")
    affected_services: List[str] = Field(default_factory=list, description="Affected services")
    affected_controllers: List[str] = Field(default_factory=list, description="Affected controllers")
    affected_repositories: List[str] = Field(default_factory=list, description="Affected repositories")
    affected_dtos: List[str] = Field(default_factory=list, description="Affected DTOs/models")
    affected_ui_components: List[str] = Field(default_factory=list, description="Affected UI components")
    affected_ui_services: List[str] = Field(default_factory=list, description="Affected UI services")
    affected_ui_stores: List[str] = Field(default_factory=list, description="Affected UI stores (state)")
    database_changes_required: bool = Field(default=False, description="DB changes needed")
    api_changes_required: bool = Field(default=False, description="API changes needed")
    requires_migration: bool = Field(default=False, description="Migration required")
    migration_type: Optional[str] = Field(None, description="Type of migration needed")


class LocalizationResult(BaseModel):
    """Result of localization phase - exact targets discovered"""
    target_files: List[TargetFile] = Field(..., description="Files to modify/create")
    target_methods: List[TargetMethod] = Field(..., description="Specific methods to modify")
    dependencies: List[str] = Field(..., description="Dependent components")
    execution_paths: List[ExecutionPath] = Field(..., description="Execution paths analyzed")
    impact_analysis: ImpactAnalysis = Field(..., description="Impact analysis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    consensus_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Consensus-adjusted localization confidence",
    )
    source_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of evidence source families used during localization",
    )
    evaluation_mode: Optional[str] = Field(
        None,
        description="Evaluation mode or feature-flag summary for the run",
    )
    requires_human_review: bool = Field(..., description="Whether human review needed")
    user_approved: bool = Field(default=False, description="Whether user approved")
    user_feedback: Optional[str] = Field(None, description="User feedback/comments")


# ============================================================================
# HUMAN APPROVAL MODELS
# ============================================================================

class ApprovalMode(str, Enum):
    """Mode for human approval"""
    ALWAYS = "always"  # Always ask for approval
    CONFIDENCE_BASED = "confidence_based"  # Ask based on confidence threshold
    NEVER = "never"  # Never ask (fully autonomous)


class ApprovalRequest(BaseModel):
    """Request for human approval"""
    phase: str = Field(..., description="Which phase (localization, planning, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    approval_level: str = Field(..., description="optional, notification, required, manual_selection")
    message: str = Field(..., description="Message for user")
    
    # What AI suggested
    suggested_files: List[TargetFile] = Field(..., description="Files AI suggests modifying")
    suggested_methods: List[TargetMethod] = Field(..., description="Methods AI suggests modifying")
    dependencies: List[str] = Field(..., description="Dependent components")
    execution_paths: List[ExecutionPath] = Field(..., description="Execution paths")
    impact_analysis: ImpactAnalysis = Field(..., description="Impact analysis")
    
    # Allow user to add/remove
    available_files: List[str] = Field(default_factory=list, description="All project files for manual selection")
    allow_manual_addition: bool = Field(default=True, description="Allow user to add files")
    allow_file_removal: bool = Field(default=True, description="Allow user to remove suggested files")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


class ApprovalResponse(BaseModel):
    """User's approval response"""
    action: str = Field(..., description="approve, cancel, request_changes")
    
    # Approved items
    approved_files: List[TargetFile] = Field(default_factory=list, description="Files user approved")
    approved_methods: List[TargetMethod] = Field(default_factory=list, description="Methods user approved")
    
    # User additions
    additional_files: List[str] = Field(default_factory=list, description="Files user added manually")
    removed_files: List[str] = Field(default_factory=list, description="Files user removed")
    
    # File modifications
    file_task_types: Dict[str, str] = Field(default_factory=dict, description="User-specified task types (CREATE/MODIFY)")
    
    # Feedback
    feedback: Optional[str] = Field(None, description="User feedback/comments")
    
    # Context
    workspace_path: str = Field(..., description="Workspace path")
    original_result: LocalizationResult = Field(..., description="Original localization result")
    
    # Metadata
    responded_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# ============================================================================
# PLANNING MODELS
# ============================================================================

class ArtifactType(str, Enum):
    """Classification of the artifact to be generated"""
    SOURCE_CODE = "source_code"
    STYLE = "style"
    CONFIG = "config"
    SCRIPT = "script"
    SQL = "sql"
    GENERIC_TEXT = "generic_text"


class CandidateRole(str, Enum):
    """
    Deterministic role classification for a candidate file.

    Assigned before planning so the planner LLM sees WHY a file is in the
    candidate pool.  Also used by grounded_understanding_node to honour
    planner IGNORE decisions — LOCK_FILE / TEST / GENERATED candidates that
    the planner did not select are never auto-promoted.
    """
    VERSION_SOURCE  = "VERSION_SOURCE"   # Java/TS constant holding the version
    VERSION_DISPLAY = "VERSION_DISPLAY"  # Template / UI that renders the version
    DEPLOYMENT      = "DEPLOYMENT"       # Helm charts, shell scripts, manifests
    CONFIG          = "CONFIG"           # JSON / YAML / properties config files
    ROOT_COMPONENT  = "ROOT_COMPONENT"   # App root (app.component.ts, App.tsx …)
    TEST            = "TEST"             # Unit / spec / integration test files
    STYLE           = "STYLE"            # SCSS / CSS / SASS style sheets
    GENERATED       = "GENERATED"        # dist/, node_modules/, target/, build/
    LOCK_FILE       = "LOCK_FILE"        # package-lock.json, yarn.lock, pom.xml
    DOCUMENTATION   = "DOCUMENTATION"   # Markdown / RST / TXT docs
    REFERENCE       = "REFERENCE"       # Domain-relevant but wrong service (read-only)
    UNKNOWN         = "UNKNOWN"          # Anything else


class PlannerDecisionValue(str, Enum):
    """
    Planner's explicit intent for an evidence-backed candidate.

    Derived automatically after planning by comparing the planner's task list
    with the candidates that were passed to it.  Stored in workflow state and
    respected by grounded_understanding_node.
    """
    REQUIRED = "REQUIRED"  # Planner created a modify/create task
    OPTIONAL = "OPTIONAL"  # Planner included as read_only context
    IGNORE   = "IGNORE"    # Planner evaluated and explicitly excluded


class TaskType(str, Enum):
    """Type of development task"""
    READ_ONLY = "read_only"   # analysis/context gathering — NO file writes
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    REFACTOR = "refactor"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages"""
    CSHARP = "csharp"
    CSHARP_TEST = "csharp_test"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    JAVA = "java"
    GO = "go"
    XML = "xml"
    JSON = "json"
    SHELL = "shell"      # .sh, .bash, .ps1, .bat — deployment/build scripts
    SCSS = "scss"        # .scss, .css, .less — stylesheets
    HTML = "html"        # .html, .htm — markup templates (Angular/Vue/JSX)
    YAML = "yaml"        # .yml, .yaml — config/manifests
    PROPERTIES = "properties"  # .properties, .env — key=value config
    SQL = "sql"          # .sql — Flyway/database migrations and DDL/DML scripts
    MARKDOWN = "markdown"
    TEXT = "text"


class LocalizationCandidate(BaseModel):
    """A single ranked candidate from the localization layer."""
    path: str = Field(..., description="Workspace-relative file path")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Combined confidence score")
    consensus_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Consensus-calibrated confidence score",
    )
    signals: List[str] = Field(
        default_factory=list,
        description="Signals that contributed to this score, e.g. ['extension_match', 'path_segment_match', 'keyword_path', 'keyword_content']"
    )
    raw_score: float = Field(
        default=0.0,
        description="Raw combined score before normalization"
    )
    source_votes: List[str] = Field(
        default_factory=list,
        description="Evidence source families that support this candidate",
    )
    workflow_guarded: bool = Field(
        default=False,
        description="Whether workflow evidence was gated by direct evidence requirements",
    )


class BlueprintStatus(str, Enum):
    """Lifecycle state of a cross-file symbol blueprint.

    LLM output is NEVER trusted as truth.  Every blueprint starts as PLANNED
    and must be deterministically verified before downstream consumption.

    Lifecycle:
        Planner LLM emits            → PLANNED  (semantic intent only)
        SymbolResolver / LSP checks  → EVIDENCE_VERIFIED  or  INVALID
        Code generator writes it     → IMPLEMENTED_CANDIDATE (not yet validated)
        Post-gen verification passes → IMPLEMENTED
        Dependency check passes      → DEPENDENCY_VALIDATED
        Build/compile passes         → BUILD_VALIDATED

    Key invariant:
        The Planner can propose a capability.  It CANNOT declare that a symbol exists.
        Only repository evidence (LSP/AST/compiler) can promote to EVIDENCE_VERIFIED.
        Only validation can promote IMPLEMENTED_CANDIDATE to IMPLEMENTED.
    """
    PLANNED = "planned"                         # Semantic intent from planner — NOT fact
    EVIDENCE_VERIFIED = "evidence_verified"     # Confirmed by SymbolResolver / LSP / AST
    INVALID = "invalid"                         # Failed verification — needs correction
    IMPLEMENTED_CANDIDATE = "implemented_candidate"  # Appeared in generated code, NOT yet validated
    IMPLEMENTED = "implemented"                 # Validation confirmed the implementation is correct
    DEPENDENCY_VALIDATED = "dependency_validated"  # Consumers verified OK
    BUILD_VALIDATED = "build_validated"          # Compilation / build passed

    # Backward-compat aliases (for existing data that uses old names)
    PROPOSED = "planned"                        # alias → PLANNED
    VERIFIED = "evidence_verified"              # alias → EVIDENCE_VERIFIED
    VALIDATED = "build_validated"               # alias → BUILD_VALIDATED


class ContextTier(str, Enum):
    """Tiered rolling context — start cheap, escalate when evidence demands it.

    The system begins at SIGNATURE and only escalates when the LLM or
    validation logic indicates that more context is needed.
    """
    SIGNATURE = "signature"              # Method/property names + types
    TYPE_CONTRACT = "type_contract"      # + return types, param types, interfaces
    IMPLEMENTATION = "implementation"    # + relevant method bodies
    FULL_FILE = "full_file"              # Complete file content


class ReferenceStatus(str, Enum):
    """Resolution status of an outbound reference found in generated code.

    Separates planner/generator intent from verified repository reality:
    a generated call is NEVER trusted just because it looks plausible.
    """
    EXISTING_VERIFIED = "existing_verified"      # Callee exists in repository truth
    GENERATED_VERIFIED = "generated_verified"    # Callee was generated+verified this run
    UNRESOLVED = "unresolved"                    # Receiver type known, member does NOT exist
    UNKNOWN_RECEIVER = "unknown_receiver"        # Receiver type could not be inferred (advisory)


class ErrorCategory(str, Enum):
    """Attribution of a build/compile error relative to the current ticket."""
    TICKET_INTRODUCED = "ticket_introduced"                  # In a file our ticket modified
    GENERATED_COMPANION = "generated_companion"              # In a file we generated this run
    GENERATED_DEPENDENCY_FAILURE = "generated_dependency_failure"  # Missing generated dependency
    PRE_EXISTING = "pre_existing"                            # In a file we never touched — do NOT fix
    INFRASTRUCTURE = "infrastructure"                        # Env/dependency/network — not a source defect
    UNKNOWN = "unknown"                                      # Cannot attribute


class SemanticBlueprint(BaseModel):
    """High-level cross-file intent from the planner.

    Contains WHAT needs to cross file boundaries, not HOW.
    Exact symbols/signatures come from evidence after generation.

    The planner fills these fields.  The code generator uses them as
    *intent guidance* while relying on repository evidence for actual
    symbol resolution.
    """
    capability: str = Field(
        ..., description="What capability crosses the boundary, e.g. 'project membership check'"
    )
    data_shape: str = Field(
        default="",
        description="Shape of the data that crosses, e.g. 'boolean + organization name'"
    )
    from_task: str = Field(
        default="",
        description="Task ID this depends on (for consumes)"
    )
    relationship_type: str = Field(
        default="data",
        description="Kind of cross-file relationship: 'data', 'trigger', 'precondition', "
                    "'consumer', 'side_effect', 'shared_type'"
    )


class VerifiedSymbol(BaseModel):
    """A symbol extracted from actual generated or existing code.

    Richer than a plain string — preserves the highest-quality evidence
    available from LSP, AST, compiler, or regex extraction.

    Used in GenerationHandoff to give downstream tasks precise, verified
    facts about what was created/modified.

    Extraction priority (from plan's Evidence Authority Hierarchy):
        1. LSP / AST          → highest quality
        2. Compiler / SymbolResolver
        3. Language-aware parser
        4. Regex fallback     → acceptable but not ultimate authority
    """
    name: str = Field(..., description="Symbol name, e.g. 'checkMembership'")
    kind: str = Field(
        default="unknown",
        description="Symbol kind: 'method', 'class', 'interface', 'property', "
                    "'function', 'type', 'enum', 'constant', 'unknown'"
    )
    owner: str = Field(
        default="",
        description="Owning class/type, e.g. 'MembersService' (empty for top-level)"
    )
    signature: str = Field(
        default="",
        description="Full signature, e.g. 'checkMembership(projectId: string): Observable<...>'"
    )
    file_path: str = Field(default="", description="File where this symbol lives")
    export_status: str = Field(
        default="unknown",
        description="'exported', 'internal', 'unknown'"
    )
    source_line: Optional[int] = Field(
        default=None,
        description="Line number in file (if available)"
    )
    extraction_method: str = Field(
        default="regex",
        description="How this symbol was extracted: 'lsp', 'ast', 'compiler', 'parser', 'regex'"
    )


class SignatureBlueprint(BaseModel):
    """A method/property/type that crosses file boundaries.

    Lifecycle:
        Planner LLM emits           → status=PROPOSED
        SymbolResolver checks        → status=VERIFIED or INVALID
        Code generator writes it     → status=IMPLEMENTED
        Post-gen verification passes → status=VALIDATED

    The Code Generator ONLY trusts VERIFIED or VALIDATED blueprints.
    PROPOSED blueprints are shown as "planned but unverified".

    The planner is allowed to express uncertainty:
        confidence=0.3, verification_required=True
    meaning: "I think this symbol should exist, but verify first."
    """
    symbol_name: str = Field(..., description="Name of the cross-file symbol")
    owner_class: str = Field(default="", description="Class/type that owns this symbol")
    file_path: str = Field(..., description="File where this symbol should exist")
    signature: str = Field(
        default="",
        description="Full signature, e.g. 'getProjectMembers(projectId: string): Observable<Member[]>'"
    )
    import_path: str = Field(
        default="",
        description="Import path for consumers, e.g. '../services/member.service'"
    )
    created_by_task: str = Field(
        default="",
        description="Task ID that produces this symbol (empty if it already exists)"
    )
    consumed_by_tasks: List[str] = Field(
        default_factory=list,
        description="Task IDs that depend on this symbol"
    )
    status: BlueprintStatus = Field(
        default=BlueprintStatus.PROPOSED,
        description="Current lifecycle state"
    )
    verification_note: str = Field(
        default="",
        description="Human-readable reason for the status (e.g. why INVALID)"
    )
    actual_symbol: str = Field(
        default="",
        description="If INVALID, the actual symbol name found in the repository"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Planner confidence. LOW (<0.5) triggers mandatory verification"
    )
    verification_required: bool = Field(
        default=False,
        description="True when planner is uncertain — forces verification before use"
    )


class CrossFileContract(BaseModel):
    """Cross-file promises for a single DevelopmentTask.

    Each task declares:
      - produces: capabilities this task will PROVIDE to other tasks
      - consumes: capabilities this task NEEDS from earlier tasks

    These are SEMANTIC INTENT from the planner — they describe WHAT crosses
    file boundaries, not the exact implementation.  Exact symbols come from
    evidence (SymbolResolver/LSP/AST) after generation.

    The produces/consumes graph defines the task dependency order and
    determines what rolling context each task needs.
    """
    produces: List[SemanticBlueprint] = Field(
        default_factory=list,
        description="Capabilities this task will provide (semantic intent)"
    )
    consumes: List[SemanticBlueprint] = Field(
        default_factory=list,
        description="Capabilities this task needs from earlier tasks (semantic intent)"
    )


class GenerationHandoff(BaseModel):
    """Verified handoff produced AFTER each successful file generation + validation.

    Separates FACTS (from tools) from EXPLANATION (from LLM) so downstream
    consumers know what to trust unconditionally vs. treat as suggestion.

    Machine facts come from deterministic extraction (LSP/AST/compiler/regex).
    AI explanation comes from task metadata — treat as lower authority.

    Key invariant from architecture:
        Only validated implementation becomes authoritative context.
        If validation fails → no authoritative handoff is created.
        how_to_consume is SUGGESTIVE — actual imports should be determined
        from repository module resolution.
    """
    task_id: str = Field(..., description="Task that produced this handoff")
    file_path: str = Field(..., description="File that was generated/modified")
    timestamp: float = Field(default_factory=time.time)

    # ── Machine facts (deterministic — from AST/regex/LSP/compiler) ──
    # These are AUTHORITATIVE — downstream tasks can trust them unconditionally.
    created_symbols: List[VerifiedSymbol] = Field(
        default_factory=list,
        description="New symbols extracted from generated code"
    )
    modified_symbols: List[VerifiedSymbol] = Field(
        default_factory=list,
        description="Symbols that changed vs. original content"
    )
    exports: List[VerifiedSymbol] = Field(
        default_factory=list,
        description="All exported symbols in the file"
    )
    imports: List[str] = Field(
        default_factory=list,
        description="All import statements in the file"
    )
    signature_blueprints: List["SignatureBlueprint"] = Field(
        default_factory=list,
        description="Evidence-verified symbol contracts"
    )
    validation_status: str = Field(
        default="pending",
        description="'clean', 'warnings', 'errors', 'pending'"
    )
    dependency_issues: List[str] = Field(
        default_factory=list,
        description="Issues found by incremental validation"
    )
    extraction_method: str = Field(
        default="regex",
        description="How machine facts were obtained: 'lsp', 'ast', 'compiler', 'parser', 'regex'"
    )

    # ── AI explanation (LLM-generated — treat as SUGGESTIVE, not fact) ──
    # Downstream tasks can use these for understanding but must NOT override
    # machine facts if they contradict.
    what_changed: str = Field(
        default="",
        description="Human-readable summary of what changed"
    )
    why: str = Field(
        default="",
        description="Why this change was made (from ticket/task description)"
    )
    decisions: List[str] = Field(
        default_factory=list,
        description="Notable implementation decisions made"
    )
    how_to_consume: str = Field(
        default="",
        description="SUGGESTIVE import path — actual imports should be determined "
                    "from repository module resolution, not this field"
    )


# ── Language-Agnostic Cross-File Relationships ───────────────────────────────

class RelationshipType(str, Enum):
    """Type of relationship between two files/symbols.

    The relationship type determines how context is resolved —
    NOT the programming language. This is the core abstraction
    that makes cross-file intelligence language-agnostic.

    No language-pair conditionals (e.g. Java→TS) should ever exist.
    Instead, the system detects which RelationshipType applies and
    resolves context through the corresponding evidence provider.
    """
    SAME_LANG_IMPORT = "same_lang_import"     # import X from './Y'
    API_CONTRACT     = "api_contract"          # REST / GraphQL / RPC boundary
    DATA_FLOW        = "data_flow"             # DTO → JSON → Interface
    SHARED_TYPE      = "shared_type"           # Both use the same type/schema
    EVENT            = "event"                 # Event emitter → listener
    CONFIGURATION    = "configuration"         # Config → consumer


class RelationshipStatus(str, Enum):
    """Lifecycle of a cross-file relationship.

    Mirrors BlueprintStatus authority hierarchy.
    The AI may NEVER promote a relationship status on its own —
    only deterministic tools (LSP, AST, compiler, schema files) can verify.

    Key invariant:
        A GenerationHandoff export alone proves a symbol EXISTS in the source.
        It does NOT prove that another file consumes it.
        Only actual repository evidence (import statements, API annotations,
        schema files) can establish the connection between producer and consumer.

    Lifecycle:
        Planner says "these files are related"   → PLANNED
        Search/RAG finds a connection            → DISCOVERED
        LLM analysis suggests connection         → INFERRED
        Actual code artifact confirms it         → VERIFIED
        Compiler/build confirms it works         → VALIDATED
    """
    PLANNED     = "planned"       # Planner says these files are related
    DISCOVERED  = "discovered"    # Search/RAG found a connection
    INFERRED    = "inferred"      # LLM analysis suggests connection
    VERIFIED    = "verified"      # Actual code artifact confirms relationship
    VALIDATED   = "validated"     # Compiler/build confirms it works


class CrossFileRelationship(BaseModel):
    """Language-agnostic relationship between two files/symbols.

    This is the core model for cross-file intelligence. The system
    does NOT care what programming languages are involved — it cares
    about the RELATIONSHIP TYPE and the VERIFIED CONTRACT.

    Hard constraints:
    - Never marks inferred relationships as verified
    - No language-pair conditionals (Java→TS, Python→Go, etc.)
    - Progressive context tiers — start with signature, escalate on demand
    - Evidence sources track exactly how the relationship was established
    """
    source_file: str = Field(..., description="File that PRODUCES the capability")
    target_file: str = Field(..., description="File that CONSUMES the capability")
    relationship_type: RelationshipType = Field(
        ..., description="What kind of relationship connects them"
    )

    # Semantic capability (what crosses the boundary)
    capability: str = Field(
        ..., description="What capability crosses the boundary, "
                         "e.g. 'check project membership'"
    )

    # ── Lifecycle (v2: proper lifecycle model) ──
    status: RelationshipStatus = Field(
        default=RelationshipStatus.PLANNED,
        description="Current lifecycle state — only deterministic tools can verify"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in this relationship (0.0-1.0)"
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="How this relationship was established: "
                    "'planner', 'symbol_resolver', 'lsp', 'ast', 'compiler', "
                    "'api_annotation', 'schema_file', 'import_statement'"
    )
    verified_at: Optional[float] = Field(
        default=None, description="Timestamp when verified (None if not yet)"
    )

    # ── Progressive context tiers (populated lazily) ──
    tier_1_signature: str = Field(
        default="",
        description="Method name + params + return type (~50 chars)"
    )
    tier_2_contract: str = Field(
        default="",
        description="+ type definitions, interfaces, DTOs (~200 chars)"
    )
    tier_3_implementation: str = Field(
        default="",
        description="+ relevant method bodies (~500 chars)"
    )
    tier_4_full_file: str = Field(
        default="",
        description="Complete file content (last resort, ~2000 chars)"
    )

    # ── Provenance ──
    source_task_id: str = Field(
        default="", description="Task that generated/modified the source file"
    )
    target_task_id: str = Field(
        default="", description="Task that needs the capability"
    )
    extraction_method: str = Field(
        default="planned",
        description="How the relationship was discovered: "
                    "'lsp', 'ast', 'symbol_resolver', 'api_annotation', "
                    "'schema_file', 'import_statement', 'regex', 'planned'"
    )


class ChangeTarget(BaseModel):
    """Generic Change Target representing an authorized, granular change block/symbol.
    
    Invariant: DISCOVERY != READ_ONLY_REFERENCE != CHANGE_TARGET != AUTHORIZED_CHANGE_TARGET.
    Only an evidence-backed requirement produces an authorized ChangeTarget.
    """
    file_path: str = Field(..., description="Path of target file")

    @field_validator("file_path", mode="before")
    @classmethod
    def coerce_file_path(cls, v: Any) -> str:
        if hasattr(v, "file_path"):
            return str(v.file_path)
        if hasattr(v, "path"):
            return str(v.path)
        if isinstance(v, dict):
            return str(v.get("file_path") or v.get("path") or "")
        return str(v or "")

    symbol: Optional[str] = Field(None, description="Target symbol name (method, class, interface, selector, key, etc.)")
    symbol_type: str = Field(
        default="file",
        description="Target type: method, function, class, interface, property, constructor, route, template_block, style_rule, config_entry, file"
    )
    start_line: Optional[int] = Field(None, description="Starting line in file (1-indexed)")
    end_line: Optional[int] = Field(None, description="Ending line in file (1-indexed)")
    surrounding_context: Optional[str] = Field(None, description="Minimal surrounding context for reference")
    reason: str = Field(default="", description="Specific reason why this target is being modified")
    modification_intent: str = Field(default="", description="Intent: add_method, modify_logic, add_property, style_update, etc.")
    authorization_source: str = Field(default="", description="Source of authorization: ticket_declared_scope, proven_root_cause, migration_plan, etc.")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of evidence verifying this change is required")
    readonly_dependencies: List[str] = Field(default_factory=list, description="Files needed as read-only reference context")
    is_authorized: bool = Field(default=False, description="Whether this ChangeTarget has passed authorization validation")


class DevelopmentTask(BaseModel):
    """Individual development task in the plan"""
    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    requirement_ids: List[str] = Field(
        default_factory=list,
        description="Associated requirement IDs (e.g. ['REQ-1', 'REQ-2']) from ticket requirements"
    )
    file_path: str = Field(..., description="Target file path")

    @model_validator(mode="before")
    @classmethod
    def normalize_task_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Normalize file_path
            fp = data.get("file_path") or data.get("path") or data.get("target_file") or data.get("file")
            if hasattr(fp, "file_path"):
                data["file_path"] = str(fp.file_path)
            elif hasattr(fp, "path"):
                data["file_path"] = str(fp.path)
            elif isinstance(fp, dict):
                data["file_path"] = str(fp.get("file_path") or fp.get("path") or "")
            elif fp is not None:
                data["file_path"] = str(fp)
            else:
                data["file_path"] = ""

            # 2. Normalize task_type
            tt = str(data.get("task_type") or "modify").strip().lower()
            mapping = {
                "edit": "modify", "update": "modify", "change": "modify",
                "add": "create", "new": "create",
                "read": "read_only", "inspect": "read_only"
            }
            valid = {"read_only", "create", "modify", "delete", "refactor"}
            data["task_type"] = mapping.get(tt, tt if tt in valid else "modify")

            # 3. Normalize id
            if not data.get("id"):
                data["id"] = "task-gen"

            # 4. Normalize title & description
            if not data.get("title"):
                data["title"] = data.get("description") or data.get("selection_reason") or f"Task {data['id']}"
            if not data.get("description"):
                data["description"] = data.get("title") or f"Execute {data['id']}"

        return data

    @field_validator("file_path", mode="before")
    @classmethod
    def coerce_file_path(cls, v: Any) -> str:
        if hasattr(v, "file_path"):
            return str(v.file_path)
        if hasattr(v, "path"):
            return str(v.path)
        if isinstance(v, dict):
            return str(v.get("file_path") or v.get("path") or "")
        return str(v or "")

    task_type: TaskType = Field(..., description="Type of change")
    artifact_type: Optional[ArtifactType] = Field(None, description="Classified artifact type")
    language: ProgrammingLanguage = Field(
        default=ProgrammingLanguage.JAVA,
        description="Programming language"
    )
    dependencies: List[str] = Field(
        default_factory=list, 
        description="Task IDs this task depends on"
    )
    estimated_complexity: int = Field(
        default=3, 
        ge=1, 
        le=5, 
        description="Complexity score (1=simple, 5=complex)"
    )
    requires_testing: bool = Field(
        default=True, 
        description="Whether this change requires new tests"
    )
    selection_reason: str = Field(
        default="",
        description="Evidence-based reasoning for why this specific file was selected for modification"
    )
    # Localization fields (populated by LocalizationAgent after planning)
    target_method: Optional[str] = Field(
        None,
        description="Primary method/function to modify (for modify tasks)"
    )
    target_class: Optional[str] = Field(
        None,
        description="Target class name to modify (populated by Localization)"
    )
    allowed_methods: List[str] = Field(
        default_factory=list,
        description="ALL methods the generator is permitted to touch — decided by Planner+Localization, NOT the generator"
    )
    new_file_creation_allowed: bool = Field(
        default=False,
        description="Explicit permission for the generator to create a new file. False by default. Only True for task_type=create."
    )
    localization_confidence: float = Field(
        default=0.0,
        description="Confidence score that localized file is correct"
    )
    localization_reason: Optional[str] = Field(
        None,
        description="Why this file was chosen or created at this location"
    )
    file_candidates: List["LocalizationCandidate"] = Field(
        default_factory=list,
        description="Ranked list of candidate files from localization; [0] is file_path, others are fallbacks"
    )
    ownership_type: Optional[str] = Field(
        None,
        description="Ownership classification from Phase 2: PRIMARY_OWNER, DISPLAY_OWNER, SUPPORTING, READ_ONLY. "
                    "Populated by generate_code_node from discovered_files features. "
                    "Used by PatchValidator for ownership-tiered change budgets."
    )
    change_role: Optional[str] = Field(
        None,
        description="Assigned change role from pre-generation authorization gate."
    )
    edit_anchors: List[dict] = Field(
        default_factory=list,
        description="Per-edit anchor instructions from planner. "
                    "Each specifies: action (add_method/modify_method/add_property), "
                    "method_name, placement (sibling_after/inside/before), "
                    "anchor_method (existing method to scope against). "
                    "Used by str_replace to resolve ambiguous old_str matches."
    )
    cross_file_contract: Optional[CrossFileContract] = Field(
        None,
        description="Cross-file symbol promises: what this task produces and consumes. "
                    "Populated by the planner; verified by BlueprintVerifier."
    )
    change_targets: List[ChangeTarget] = Field(
        default_factory=list,
        description="Granular, evidence-backed change targets for this task. Drives target-first code generation."
    )
    modification_proof: Optional[str] = Field(
        default=None,
        description="Proof of required modification for dependency tasks"
    )
    change_intent: Optional[Any] = Field(
        default=None,
        description="System-computed change intent authorization"
    )
    model_config = ConfigDict(extra="allow")


class ArchitecturalPattern(str, Enum):
    """Supported architectural patterns"""
    MVC = "MVC"
    MVVM = "MVVM"
    FEATURE_SLICED = "Feature-Sliced Design"
    LAYERED = "Layered Architecture"
    MICROSERVICES = "Microservices"
    CLEAN_ARCHITECTURE = "Clean Architecture"
    MONOLITHIC = "Monolithic"


class APIChange(BaseModel):
    """API endpoint change"""
    endpoint: str = Field(..., description="API endpoint path")
    method: str = Field(..., description="HTTP method (GET, POST, etc.)")
    description: str = Field(..., description="What this endpoint does")
    required: bool = Field(
        default=True, 
        description="Whether this API is required for feature"
    )
    request_body: Optional[Dict[str, Any]] = Field(
        None, 
        description="Request body schema"
    )
    response_body: Optional[Dict[str, Any]] = Field(
        None, 
        description="Response body schema"
    )


class DatabaseChange(BaseModel):
    """Database schema change"""
    table_name: str = Field(..., description="Table name")
    change_type: str = Field(..., description="CREATE, ALTER, DROP")
    migration_script: Optional[str] = Field(
        None, 
        description="SQL migration script"
    )


class ArchitecturalPlan(BaseModel):
    """High-level architectural plan for implementation"""
    pattern: ArchitecturalPattern = Field(
        default=ArchitecturalPattern.LAYERED, 
        description="Architectural pattern to follow"
    )
    rationale: str = Field(
        default="",
        description="Explanation of the overall strategy: what problem is being solved, why this specific shape of solution (e.g. why these specific files) was chosen, and how the tasks fit together."
    )
    affected_modules: List[str] = Field(
        default_factory=list, 
        description="Modules/layers that will be modified"
    )
    tasks: List[DevelopmentTask] = Field(
        ..., 
        description="Ordered list of development tasks"
    )
    api_changes: List[APIChange] = Field(
        default_factory=list, 
        description="Required API changes"
    )
    database_changes: List[DatabaseChange] = Field(
        default_factory=list, 
        description="Database schema changes"
    )
    external_dependencies: List[str] = Field(
        default_factory=list,
        description="External packages/libraries needed"
    )
    estimated_total_complexity: int = Field(
        default=0, 
        description="Total complexity score"
    )
    signature_blueprints: List[SignatureBlueprint] = Field(
        default_factory=list,
        description="Global list of cross-file symbol blueprints across all tasks. "
                    "Union of all task-level produces. Verified by BlueprintVerifier "
                    "after planning, before code generation."
    )

class ValidatedTask(BaseModel):
    """A highly constrained task passed to the Code Generator"""
    file_path: str = Field(..., description="Target file path")
    task_type: TaskType = Field(..., description="Type of change (create, modify, delete)")
    allowed_change_surfaces: List[str] = Field(
        default_factory=list, 
        description="Symbols/methods the generator is explicitly permitted to modify"
    )
    capability_context: str = Field(
        default="", 
        description="Capability mapping justification"
    )

class ValidatedPlan(BaseModel):
    """The deterministic output of the Plan Consistency Validator"""
    tasks: List[ValidatedTask] = Field(..., description="Constrained list of tasks")
    pattern: ArchitecturalPattern = Field(..., description="Architectural pattern to follow")


# ============================================================================
# RAG & CONTEXT MODELS
# ============================================================================

class CodeChunkType(str, Enum):
    """Type of code chunk"""
    CLASS = "class"
    CLASS_SUMMARY = "class_summary"
    FUNCTION = "function"
    INTERFACE = "interface"
    ENUM = "enum"
    MODULE = "module"
    COMPONENT = "component"


class CodeChunk(BaseModel):
    """Code chunk retrieved from vector store"""
    content: str = Field(..., description="Code content")
    file_path: str = Field(..., description="Source file path")
    chunk_type: CodeChunkType = Field(..., description="Type of code element")
    name: str = Field(..., description="Name of class/function/etc.")
    namespace: Optional[str] = Field(
        None, 
        description="Namespace or module path"
    )
    dependencies: List[str] = Field(
        default_factory=list, 
        description="Dependencies this code uses"
    )
    similarity_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Similarity score from vector search"
    )
    line_start: Optional[int] = Field(None, description="Starting line number")
    line_end: Optional[int] = Field(None, description="Ending line number")


class ContextEvaluation(BaseModel):
    """Evaluation of whether retrieved context is sufficient"""
    sufficient: bool = Field(
        ..., 
        description="Is context sufficient for code generation?"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence in this evaluation"
    )
    missing_elements: List[str] = Field(
        default_factory=list, 
        description="Missing classes/functions/modules"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving context"
    )
    retrieval_strategy: Optional[str] = Field(
        None,
        description="Suggested retrieval strategy if insufficient"
    )


# ============================================================================
# CODE GENERATION MODELS
# ============================================================================

class GeneratedCode(BaseModel):
    """Generated code output"""
    file_path: str = Field(..., description="Target file path")
    content: str = Field(..., description="Generated code content")
    language: ProgrammingLanguage = Field(..., description="Programming language")
    change_type: TaskType = Field(..., description="Type of change")
    documentation: Optional[str] = Field(
        None, 
        description="Generated documentation/comments"
    )
    test_file_path: Optional[str] = Field(
        None, 
        description="Path to corresponding test file"
    )
    imports: List[str] = Field(
        default_factory=list,
        description="Required imports/using statements"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Generation timestamp"
    )


class GeneratedTests(BaseModel):
    """Generated test suite"""
    test_file_path: str = Field(..., description="Test file path")
    content: str = Field(..., description="Test code content")
    test_framework: str = Field(
        default="MSTest", 
        description="Test framework (MSTest, xUnit, NUnit, etc.)"
    )
    test_cases: List[str] = Field(
        default_factory=list,
        description="List of test case names"
    )
    coverage_estimate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated code coverage percentage"
    )


# ============================================================================
# EXECUTION MODELS
# ============================================================================

class BuildStatus(str, Enum):
    """Build execution status"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"  # No compiler was run (nothing buildable / no manifest). NOT a verified success.


class BuildResult(BaseModel):
    """Build execution result"""
    status: BuildStatus = Field(..., description="Build status")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    exit_code: int = Field(..., description="Process exit code")
    errors: List[str] = Field(
        default_factory=list, 
        description="Extracted error messages"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Extracted warning messages"
    )
    duration_seconds: float = Field(
        default=0.0, 
        description="Build duration in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Execution timestamp"
    )


class CompilerError(BaseModel):
    """Structured representation of a single compiler error"""
    file_path: str
    error_code: str
    message: str


class BuildAttributionResult(BaseModel):
    """Result of analyzing build errors against modified files"""
    build_passed: bool
    ai_introduced_errors: List[CompilerError] = Field(default_factory=list)
    pre_existing_errors: List[CompilerError] = Field(default_factory=list)
    modified_files: List[str] = Field(default_factory=list)
    affected_modified_files: List[str] = Field(default_factory=list)
    unaffected_modified_files: List[str] = Field(default_factory=list)
    should_retry_build_fix: bool = Field(default=False)
    repository_is_dirty: bool = Field(default=False)
    reasoning: str = Field(default="")


class ContextHealth(BaseModel):
    """Health metrics for context retrieval to inform the planner"""
    rag_results: int = 0
    neo4j_results: int = 0
    sqlite_results: int = 0
    grep_results: int = 0
    overall: str = Field(default="UNKNOWN", description="HEALTHY, DEGRADED, or CRITICAL")



class TestStatus(str, Enum):
    """Test execution status"""
    ALL_PASSED = "all_passed"
    SUCCESS = "all_passed"
    NO_TESTS = "no_tests"
    SKIPPED = "skipped"
    SOME_FAILED = "some_failed"
    ALL_FAILED = "all_failed"
    FAILED = "some_failed"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestResult(BaseModel):
    """Test execution result"""
    status: TestStatus = Field(..., description="Overall test status")
    total_tests: int = Field(default=0, description="Total number of tests")
    passed: int = Field(default=0, description="Number of passed tests")
    failed: int = Field(default=0, description="Number of failed tests")
    skipped: int = Field(default=0, description="Number of skipped tests")
    errors: List[str] = Field(
        default_factory=list, 
        description="Test failure messages"
    )
    test_output: str = Field(default="", description="Full test output")
    duration_seconds: float = Field(
        default=0.0, 
        description="Test duration in seconds"
    )
    coverage_percentage: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Code coverage percentage"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Execution timestamp"
    )
    
    @property
    def success(self) -> bool:
        """Convenience property for checking success"""
        return self.status == TestStatus.ALL_PASSED


class ExecutionError(BaseModel):
    """Execution error details for debugging"""
    error_type: str = Field(..., description="Type of error (compile, runtime, test)")
    message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    file_path: Optional[str] = Field(None, description="File where error occurred")
    line_number: Optional[int] = Field(None, description="Line number of error")
    code_snippet: Optional[str] = Field(
        None, 
        description="Code snippet around error"
    )


# ============================================================================
# DEBUG MODELS
# ============================================================================

class ErrorAnalysis(BaseModel):
    """Error analysis from debug agent"""
    root_cause: str = Field(..., description="Identified root cause")
    affected_lines: List[int] = Field(
        default_factory=list,
        description="Line numbers affected"
    )
    fix_priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Priority of fixing (1=highest)"
    )


class FixedCode(BaseModel):
    """Fixed code after debugging iteration"""
    original_content: str = Field(..., description="Original code content")
    fixed_content: str = Field(..., description="Fixed code content")
    changes_made: List[str] = Field(
        ..., 
        description="List of changes applied"
    )
    iteration: int = Field(..., description="Debug iteration number")
    success_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated probability of fix success"
    )


# ============================================================================
# INVESTIGATION + EVIDENCE COLLECTION MODELS
# ============================================================================

class InvestigationHypothesis(BaseModel):
    """
    A single LLM-generated hypothesis about the root cause or change area.

    The Investigation Agent produces a list of these; the Evidence Collection
    Loop uses them as seeds for every query it fires.
    """
    id: str = Field(..., description="Stable identifier, e.g. 'H1', 'H2'")
    hypothesis: str = Field(..., description="Plain-language hypothesis statement")
    queries: List[str] = Field(
        default_factory=list,
        description="Semantic-search queries derived from this hypothesis"
    )
    literals: List[str] = Field(
        default_factory=list,
        description="Exact string / constant literals to search in the codebase"
    )
    symbols: List[str] = Field(
        default_factory=list,
        description="Class / method / component / config-key names to investigate"
    )
    core_subjects: List[str] = Field(
        default_factory=list,
        description="Core entities/values being changed (e.g., 'version number', 'deliverable count')"
    )
    context_locations: List[str] = Field(
        default_factory=list,
        description="Contextual locations where the issue manifests (e.g., 'dashboard', 'CC4E app')"
    )
    macro_directories: List[str] = Field(
        default_factory=list,
        description="Target top-level folders mapped by the Macro Architecture Brain"
    )
    anchors: List[str] = Field(
        default_factory=list,
        description="Domain/page/business nouns extracted from the ticket to anchor the component search"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Prior confidence before evidence is collected"
    )
    business_intent: Optional[str] = Field(
        default=None,
        description="High-level business intent of this hypothesis (e.g., 'version bump', 'css fix'). Used by Graphify provider to query by concept rather than literal."
    )


class EvidenceItem(BaseModel):
    """
    A single piece of evidence collected during the Evidence Collection Loop.
    """
    file_path: str = Field(..., description="Repo-relative path of the evidence file")

    @field_validator("file_path", mode="before")
    @classmethod
    def coerce_file_path(cls, v: Any) -> str:
        if hasattr(v, "file_path"):
            return str(v.file_path)
        if hasattr(v, "path"):
            return str(v.path)
        if isinstance(v, dict):
            return str(v.get("file_path") or v.get("path") or "")
        return str(v or "")

    provider: str = Field(default="", description="E.g., 'literal', 'alias', 'concept', 'api', 'graph', 'vector'")
    source: str = Field(default="", description="Alias for provider/source of evidence")
    strength: str = Field(..., description="'strong', 'medium', 'weak'")
    details: str = Field(default="", description="Specific details, e.g. '260200 matched' or 'Navigation Bar alias'")
    content_snippet: str = Field(
        default="",
        description="Relevant code / text snippet (truncated to ~400 chars)"
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,

        description="How relevant this piece of evidence is to the hypothesis"
    )
    hypothesis_id: Optional[str] = Field(
        None,
        description="Which InvestigationHypothesis this evidence supports (or None if general)"
    )
    symbol_name: Optional[str] = Field(
        None,
        description="Specific symbol / class / method name found in this evidence"
    )
    
    # ── Graph/Relationship specific fields ──────────────────────────────────────
    graph_distance: Optional[int] = Field(
        None,
        description="Distance in edges from the matched concept/literal to this file"
    )
    matched_concept: Optional[str] = Field(
        None,
        description="The business concept or literal that triggered this discovery"
    )
    path: Optional[List[Dict[str, str]]] = Field(
        None,
        description="List of nodes in the graph traversal path (e.g., [{'type': 'Capability', 'id': '...'}])"
    )
    confidence_reason: Optional[str] = Field(
        None,
        description="A human-readable reason why this evidence was selected (e.g., 'Direct capability relationship')"
    )
    provenance: Optional[Dict[str, str]] = Field(
        None,
        description="Metadata about the source (e.g., graph_version, repository_commit)"
    )

    @model_validator(mode='before')
    @classmethod
    def sync_source_and_provider(cls, data: Any) -> Any:
        if isinstance(data, dict):
            provider = data.get("provider", "")
            source = data.get("source", "")
            if provider and not source:
                data["source"] = provider
            elif source and not provider:
                data["provider"] = source
        return data


class ComponentGroup(BaseModel):
    """
    A group of framework-related files that form a single component.

    In Angular, a component is typically:
        add-members.component.ts + .html + .scss
    In React:
        AddMembers.tsx + AddMembers.module.css

    Standalone files (Java services, config files, etc.) are represented
    as groups of 1 with is_group=False — they receive ZERO penalty or
    advantage from the grouping system.

    Usage::

        group = ComponentGroup(
            stem="add-members.component",
            directory="src/app/modules/members/add-members",
            members={".ts": "src/.../add-members.component.ts",
                     ".html": "src/.../add-members.component.html"},
            primary_file="src/.../add-members.component.ts",
            group_score=0.85,
            evidence_sources=[".ts"],
        )
    """
    stem: str = Field(
        ..., description="Component stem, e.g. 'add-members.component'"
    )
    directory: str = Field(
        ..., description="Directory containing the component files"
    )
    members: Dict[str, str] = Field(
        ...,
        description="Mapping of extension → repo-relative file path. "
                    "E.g. {'.ts': 'src/.../foo.ts', '.html': 'src/.../foo.html'}"
    )
    primary_file: str = Field(
        ...,
        description="The member file with the highest evidence score "
                    "(the one originally found by evidence collection)"
    )
    group_score: float = Field(
        default=0.0,
        description="Maximum evidence score across all members with evidence"
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Extensions of members that had actual evidence "
                    "(e.g. ['.ts'] means only .ts was found by evidence)"
    )
    is_group: bool = Field(
        default=True,
        description="True if the group has >1 member. False for standalone files "
                    "(no siblings found). Standalone files get ZERO penalty or "
                    "advantage from the grouping system."
    )

    def all_file_paths(self) -> List[str]:
        """Return all member file paths."""
        return list(self.members.values())

    def non_evidence_members(self) -> Dict[str, str]:
        """Return members that were discovered via disk scan (no evidence)."""
        return {
            ext: path for ext, path in self.members.items()
            if ext not in self.evidence_sources
        }

    def to_dict(self) -> dict:
        return {
            "stem": self.stem,
            "directory": self.directory,
            "members": self.members,
            "primary_file": self.primary_file,
            "group_score": round(self.group_score, 4),
            "evidence_sources": self.evidence_sources,
            "is_group": self.is_group,
        }


class GraphQuery(BaseModel):
    """
    Rich query object used by RelationshipProvider to find interconnected files.
    """
    literals: List[str] = Field(default_factory=list)
    symbols: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)
    routes: List[str] = Field(default_factory=list)
    configs: List[str] = Field(default_factory=list)


class GroundedUnderstanding(BaseModel):
    """
    The synthesized understanding produced AFTER the Evidence Collection Loop.

    Passed into Generation so the LLM operates on verified evidence rather
    than only the localized file list.
    """
    root_cause: str = Field(
        ...,
        description="Concise statement of the confirmed root cause"
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="All collected evidence items"
    )
    dependency_chain: List[str] = Field(
        default_factory=list,
        description="Ordered list of file/component paths from entry point to change site"
    )
    change_group: List[str] = Field(
        default_factory=list,
        description="Files that must change together to fully resolve the ticket"
    )
    required_files: List[str] = Field(
        default_factory=list,
        description="All files needed (writable + context)"
    )
    writable_files: List[str] = Field(
        default_factory=list,
        description="Files the generator is permitted to modify"
    )
    readonly_context_files: List[str] = Field(
        default_factory=list,
        description="Read-only files needed as context"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall evidence confidence after the loop"
    )
    evidence_loop_iterations: int = Field(
        default=0,
        description="Number of evidence collection iterations performed"
    )
    hypotheses_confirmed: List[str] = Field(
        default_factory=list,
        description="IDs of hypotheses supported by collected evidence"
    )
    hypotheses_refuted: List[str] = Field(
        default_factory=list,
        description="IDs of hypotheses contradicted or unsupported by evidence"
    )


# ============================================================================
# WORKFLOW MODELS
# ============================================================================

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    INITIALIZED = "initialized"
    INVESTIGATING = "investigating"  # NEW: Initial investigation phase
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RETRIEVING_CONTEXT = "retrieving_context"
    EVALUATING_CONTEXT = "evaluating_context"
    GENERATING_CODE = "generating_code"
    GENERATING_TESTS = "generating_tests"
    EXECUTING_BUILD = "executing_build"
    EXECUTING_TESTS = "executing_tests"
    DEBUGGING = "debugging"
    INTEGRATING = "integrating"
    COMPLETED = "completed"
    EXPLANATION_PROVIDED = "explanation_provided"  # NEW: No code needed, just explanation
    FAILED = "failed"
    CANCELLED = "cancelled"


class IntegrationStatus(str, Enum):
    """Integration status"""
    NOT_STARTED = "not_started"
    BRANCH_CREATED = "branch_created"
    CHANGES_COMMITTED = "changes_committed"
    PR_CREATED = "pr_created"
    TICKET_UPDATED = "ticket_updated"
    COMPLETED = "completed"
    FAILED = "failed"


class IntegrationResult(BaseModel):
    """Result of integration step"""
    status: IntegrationStatus = Field(..., description="Integration status")
    branch_name: Optional[str] = Field(None, description="Git branch name")
    commit_sha: Optional[str] = Field(None, description="Commit SHA")
    pull_request_url: Optional[str] = Field(None, description="PR URL")
    valueedge_updated: bool = Field(
        default=False,
        description="Whether ValueEdge ticket was updated"
    )
    errors: List[str] = Field(default_factory=list, description="Any errors")


class AutonomousWorkflowState(BaseModel):
    """Complete workflow state for autonomous execution"""
    # Input
    ticket: ValueEdgeTicket = Field(..., description="Input ticket")
    
    # Status
    status: WorkflowStatus = Field(
        default=WorkflowStatus.INITIALIZED,
        description="Current workflow status"
    )
    
    # Investigation Phase (NEW)
    investigation_result: Optional[InvestigationResult] = Field(
        None,
        description="Initial investigation/triage result"
    )
    
    # Analysis Phase
    requirements: Optional[StructuredRequirements] = Field(
        None,
        description="Analyzed requirements"
    )
    
    # Planning Phase
    plan: Optional[ArchitecturalPlan] = Field(
        None,
        description="Architectural plan"
    )
    
    # Context Retrieval Phase
    context: List[CodeChunk] = Field(
        default_factory=list,
        description="Retrieved code context"
    )
    context_evaluation: Optional[ContextEvaluation] = Field(
        None,
        description="Context evaluation result"
    )
    
    # Generation Phase
    generated_files: List[GeneratedCode] = Field(
        default_factory=list,
        description="Generated code files"
    )
    generated_tests: Optional[GeneratedTests] = Field(
        None,
        description="Generated test suite"
    )
    
    # Execution Phase
    build_result: Optional[BuildResult] = Field(
        None,
        description="Build execution result"
    )
    test_result: Optional[TestResult] = Field(
        None,
        description="Test execution result"
    )
    
    # Debug Phase
    debug_iterations: int = Field(
        default=0,
        description="Number of debug iterations"
    )
    error_analyses: List[ErrorAnalysis] = Field(
        default_factory=list,
        description="Error analyses from debug iterations"
    )
    
    # Integration Phase
    integration_result: Optional[IntegrationResult] = Field(
        None,
        description="Integration result"
    )
    
    # Tracking
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during workflow"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings during workflow"
    )
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Workflow start timestamp"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="Workflow end timestamp"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Workflow creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )
    
    def update_status(self, new_status: WorkflowStatus):
        """Update workflow status and timestamp"""
        self.status = new_status
        self.updated_at = datetime.now()
        
        if new_status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            self.completed_at = datetime.now()
    
    def add_error(self, error: str):
        """Add error to tracking list"""
        self.errors.append(error)
        self.updated_at = datetime.now()
    
    def add_warning(self, warning: str):
        """Add warning to tracking list"""
        self.warnings.append(warning)
        self.updated_at = datetime.now()
    
    def get_summary(self) -> str:
        """Generate workflow summary"""
        duration = "In Progress"
        if self.completed_at:
            delta = self.completed_at - self.created_at
            duration = f"{delta.total_seconds():.1f}s"
        
        return f"""
Autonomous Workflow Summary
===========================
Ticket: {self.ticket.ticket_id}
Status: {self.status.value}
Duration: {duration}

Progress:
- Requirements Analyzed: {'✓' if self.requirements else '✗'}
- Plan Created: {'✓' if self.plan else '✗'}
- Context Retrieved: {'✓' if self.context else '✗'}
- Code Generated: {'✓' if self.generated_files else '✗'}
- Tests Generated: {'✓' if self.generated_tests else '✗'}
- Build Executed: {'✓' if self.build_result else '✗'}
- Tests Executed: {'✓' if self.test_result else '✗'}
- Integrated: {'✓' if self.integration_result else '✗'}

Debug Iterations: {self.debug_iterations}
Errors: {len(self.errors)}
Warnings: {len(self.warnings)}
        """.strip()


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class AutonomousTicketRequest(BaseModel):
    """API request to process a ticket autonomously"""
    ticket: ValueEdgeTicket = Field(..., description="Ticket to process")
    workspace_path: str = Field(..., description="Path to code workspace")
    dry_run: bool = Field(
        default=False,
        description="If True, generate plan only without execution"
    )
    auto_integrate: bool = Field(
        default=True,
        description="If True, automatically create PR and update ticket"
    )
    max_debug_iterations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum debug iterations before giving up"
    )


class AutonomousTicketResponse(BaseModel):
    """API response for autonomous ticket processing"""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    status: WorkflowStatus = Field(..., description="Current status")
    state: AutonomousWorkflowState = Field(..., description="Complete state")
    summary: str = Field(..., description="Human-readable summary")
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps if manual intervention needed"
    )

# ============================================================================
# CONVERSATION-TO-CODE MODELS
# ============================================================================

class MessageRole(str, Enum):
    USER = "user"
    SYSTEM = "system"
    DEVELOPER = "developer"
    PLANNER = "planner"
    ASSISTANT = "assistant"

class ConversationMessage(BaseModel):
    role: MessageRole
    text: str
    images: List[str] = Field(default_factory=list, description="Base64 or URLs to screenshots")
    files: List[str] = Field(default_factory=list, description="Attached file paths")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ConversationContext(BaseModel):
    current_goal: str = Field(default="")
    known_decisions: List[str] = Field(default_factory=list)
    rejected_decisions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    expected_ui: str = Field(default="", description="Delta understanding from images")
    extracted_requirements: List[str] = Field(default_factory=list)

class ConversationState(BaseModel):
    messages: List[ConversationMessage] = Field(default_factory=list)
    context: ConversationContext = Field(default_factory=ConversationContext)
    # The generated ticket to bridge to the rest of the pipeline
    synthesized_ticket: Optional[ValueEdgeTicket] = None


# ============================================================================
# P0: PROGRESSIVE CONTEXT BUILDING — SUFFICIENCY CHECK
# ============================================================================

class SufficiencyCheck(BaseModel):
    """
    LLM-produced assessment: 'Do I know enough to proceed?'

    Used after each evidence collection iteration.
    The LLM reasons point-by-point about what is known vs missing.
    """
    is_sufficient: bool = Field(
        ..., description="True if evidence is sufficient to proceed to planning"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall confidence in current evidence (0.0–1.0)"
    )
    evidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Quality score of collected evidence items"
    )
    coverage_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How much of the ticket's scope is covered by evidence"
    )
    reasoning_points: List[str] = Field(
        ..., description="Point-by-point reasoning for the sufficiency decision"
    )
    what_is_known: List[str] = Field(
        default_factory=list,
        description="Points listing what evidence already confirms"
    )
    what_is_missing: List[str] = Field(
        default_factory=list,
        description="Points listing what evidence is still needed"
    )
    new_search_directions: List[str] = Field(
        default_factory=list,
        description="Updated hypotheses/queries for the next search iteration"
    )

    @field_validator('reasoning_points', 'what_is_known', 'what_is_missing', 'new_search_directions', mode='before')
    @classmethod
    def _coerce_to_string_list(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            return [" | ".join(f"{k}: {val}" for k, val in v.items() if val and val != "N/A")]
        if isinstance(v, list):
            res = []
            for item in v:
                if isinstance(item, str):
                    res.append(item)
                elif isinstance(item, dict):
                    res.append(" | ".join(f"{k}: {val}" for k, val in item.items() if val and val != "N/A"))
                else:
                    res.append(str(item))
            return res
        return [str(v)]

# ============================================================================
# PLANNING RECOVERY
# ============================================================================

class PlanningRecoveryAction(BaseModel):
    """LLM-produced diagnosis of why planning failed + structured recovery action.

    Generated by planning_recovery_node when candidate validation produces
    0 writable tasks.  The recovery_type determines where the workflow routes
    next (back to discover, re-plan, or terminal).

    ``genuinely_unrecoverable`` requires the LLM to establish a high evidence
    bar — planner uncertainty alone is never sufficient for terminal failure.
    """
    recovery_type: str = Field(
        ...,
        description=(
            "One of: evidence_incomplete, evidence_contaminated, "
            "wrong_candidates, requirement_ambiguous, already_implemented, "
            "genuinely_unrecoverable"
        ),
    )

    reason: str = Field(
        ..., description="Human-readable diagnosis of the failure"
    )

    # For evidence_incomplete: what specific evidence is missing
    investigation_target: Optional[str] = Field(
        None, description="Specific file or component to investigate"
    )
    investigation_query: Optional[str] = Field(
        None, description="Specific query to search for in the target"
    )

    # For wrong_candidates: what to look for instead
    alternative_search_terms: List[str] = Field(
        default_factory=list,
        description="Search terms for alternative candidate discovery"
    )

    # For evidence_contaminated: which entries had infrastructure failures
    contaminated_entries: List[str] = Field(
        default_factory=list,
        description="File paths of evidence entries with infrastructure failures"
    )

    reasoning_points: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning that led to the diagnosis"
    )

    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence in the diagnosis (0.0–1.0)"
    )

    # Recovery versioning: tracks which recovery cycle this action belongs to.
    # Consumers set status to 'consumed' after processing.
    recovery_id: Optional[str] = Field(
        None, description="Unique identifier for this recovery action (e.g. R1, R2)"
    )
    status: str = Field(
        default="active",
        description="Lifecycle status: 'active' → 'consumed'"
    )


# ============================================================================
# P0: PLAN SELF-REVIEW
# ============================================================================

class PlanReviewResult(BaseModel):
    """
    Self-review result produced by the planner after generating an ArchitecturalPlan.

    The planner critiques its own plan point-by-point before returning.
    If approved=False, the feedback is used to regenerate the plan (max 2 retries).
    """
    approved: bool = Field(
        ..., description="True if the plan passes self-review"
    )
    review_points: List[str] = Field(
        ..., description="Point-by-point review findings"
    )
    requirements_coverage: List[str] = Field(
        default_factory=list,
        description="Which functional requirements are addressed (point per requirement)"
    )
    missing_coverage: List[str] = Field(
        default_factory=list,
        description="Which requirements are NOT addressed by the plan"
    )
    evidence_alignment: List[str] = Field(
        default_factory=list,
        description="Points showing whether selected files are backed by evidence"
    )
    scope_assessment: str = Field(
        default="",
        description="Is the plan too narrow, too broad, or appropriate?"
    )
    rejection_reason: Optional[str] = Field(
        None, description="If rejected: concise reason for the rejection"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Specific suggestions to improve the plan on next attempt"
    )


# ============================================================================
# P0: MULTI-ANGLE CODE REVIEW
# ============================================================================

class ReviewFinding(BaseModel):
    """A single finding from one review angle."""
    severity: str = Field(
        ..., description="'critical', 'warning', or 'info'"
    )
    file_path: str = Field(..., description="File where the issue was found")
    description: str = Field(..., description="What is wrong")
    suggestion: str = Field(default="", description="How to fix it")


class MultiAngleReviewResult(BaseModel):
    """
    Result of 3 sequential review angles after code generation.

    1. Architecture alignment — does code match the plan?
    2. Ownership boundaries — only planned files modified?
    3. Cross-file consistency — imports, interfaces, call sites correct?

    Each angle produces findings. If any critical finding exists,
    the code is regenerated (max 1 retry).
    """
    approved: bool = Field(
        ..., description="True if no critical findings"
    )
    architecture_findings: List[ReviewFinding] = Field(
        default_factory=list,
        description="Findings from architecture alignment check"
    )
    ownership_findings: List[ReviewFinding] = Field(
        default_factory=list,
        description="Findings from ownership boundary check"
    )
    crossfile_findings: List[ReviewFinding] = Field(
        default_factory=list,
        description="Findings from cross-file consistency check"
    )
    summary_points: List[str] = Field(
        default_factory=list,
        description="Point-by-point summary of the review"
    )
    critical_count: int = Field(
        default=0, description="Number of critical findings"
    )


# ============================================================================
# P0: CONTEXT COMPACTION
# ============================================================================

class ContextSummary(BaseModel):
    """
    Compressed context summary for phase transitions.

    Used before expensive LLM calls (planning, code generation) to reduce
    context size while preserving essential information.
    """
    ticket_summary: str = Field(
        ..., description="1-2 sentence summary of the ticket"
    )
    key_requirements: List[str] = Field(
        default_factory=list,
        description="Top functional requirements (max 7)"
    )
    localized_files: List[str] = Field(
        default_factory=list,
        description="Files confirmed for modification"
    )
    evidence_signals: List[str] = Field(
        default_factory=list,
        description="Top evidence findings supporting the localization"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="What NOT to do (blacklisted files, scope limits)"
    )
    hypothesis_conclusions: List[str] = Field(
        default_factory=list,
        description="Point-by-point conclusions from hypothesis investigation"
    )


class NodeExplanation(BaseModel):
    """Dual-contract explanation emitted by each major workflow node."""
    node: str = Field(..., description="Node name")
    decision: str = Field(..., description="Primary node decision")
    rationale_points: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence_used: List[str] = Field(default_factory=list)
    evidence_rejected: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    advice_for_next: str = Field(default="")
    open_questions: List[str] = Field(default_factory=list)


class ExecutionNarrative(BaseModel):
    """Final user-facing engineering narrative from the run."""
    executive_summary: str = Field(default="")
    step_by_step: List[str] = Field(default_factory=list)
    change_ledger: List[str] = Field(default_factory=list)
    validation_ledger: List[str] = Field(default_factory=list)
    unresolved_risks: List[str] = Field(default_factory=list)


# ============================================================================
# P0: GROUNDED IMPLEMENTATION DECISION
# ============================================================================

class FileEditStrategy(str, Enum):
    """Edit strategy assigned per-file by the Grounded Resolution node."""
    DIRECT_OWNER = "DIRECT_OWNER"          # Canonical source — MUST be edited
    VERIFICATION_TARGET = "VERIFICATION_TARGET"  # Downstream — verify after owner change, don't modify
    SUPPLEMENTARY = "SUPPLEMENTARY"        # Supporting file — edit only if needed for consistency
    BLOCKED = "BLOCKED"                    # Useful reference but MUST NEVER be modified (generated, lock, swagger)
    READ_ONLY = "READ_ONLY"                # Irrelevant reference file — do NOT modify
    UNKNOWN = "UNKNOWN"                    # Could not determine role


class DeterministicSignals(BaseModel):
    """
    Pre-computed deterministic signals for a single file.

    These are gathered BEFORE the LLM is called, so the LLM reasons
    on top of objective facts instead of inventing ownership.
    """
    file_path: str
    evidence_score: float = Field(default=0.0, description="Best evidence relevance score")
    evidence_count: int = Field(default=0, description="Number of evidence items for this file")
    has_literal_match: bool = Field(default=False, description="File content contains the literal search term")
    is_config_file: bool = Field(default=False, description="File is a config/properties/yaml/env file")
    is_ui_component: bool = Field(default=False, description="File is a UI template or component")
    is_test_file: bool = Field(default=False, description="File is a test/spec file")
    is_style_file: bool = Field(default=False, description="File is a CSS/SCSS/style file")
    is_blocked: bool = Field(default=False, description="File is generated, lock, or swagger file (never modify)")
    incoming_ref_count: int = Field(default=0, description="Number of files that import/reference this file")
    outgoing_ref_count: int = Field(default=0, description="Number of files this file imports/references")
    defines_symbol: bool = Field(default=False, description="File defines (not just uses) the target symbol")
    semantic_verification_decision: str = Field(default="UNKNOWN", description="ACCEPT/REJECT from semantic verification")
    snippet_preview: str = Field(default="", description="First relevant code snippet (≤120 chars)")


class FileResolution(BaseModel):
    """Per-file resolution from the Grounded Implementation Decision."""
    file_path: str = Field(..., description="Relative file path")
    strategy: FileEditStrategy = Field(..., description="Edit strategy for this file")
    reasoning: str = Field(default="", description="Why this strategy was assigned")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs/paths of evidence items that support this classification"
    )


class PropagationExpectation(BaseModel):
    """Answers: If I modify the owner, what is expected to change automatically?"""
    source: str = Field(..., description="The owner file being modified")
    target: str = Field(..., description="The downstream consumer file")
    expectation: str = Field(..., description="What should happen automatically (e.g. 'Version displayed should update')")
    requires_code_change: bool = Field(default=False, description="Should be False for automatic propagation")


class GroundedImplementationDecision(BaseModel):
    """
    General-purpose pre-planning reasoning stage.

    Determines the implementation strategy by identifying authoritative owners,
    verification targets, supporting files, and ignored files.

    The LLM reasons ON TOP of deterministic signals (evidence scores,
    config detection, reference counts, literal matches) — it classifies
    and explains, it does NOT invent ownership.

    Works for: version changes, config changes, constants, feature flags,
    API URLs, CSS/theme changes, resource paths, database config, UI text, etc.
    """
    # ── Pre-computed categories (downstream nodes consume these directly) ──
    modifiable_files: List[str] = Field(
        default_factory=list,
        description="Files that MUST be modified (owners + supplementary that need changes)"
    )
    verification_targets: List[str] = Field(
        default_factory=list,
        description="Files to VERIFY after owner changes propagate — do NOT modify"
    )
    supporting_files: List[str] = Field(
        default_factory=list,
        description="Files needed for consistency (imports, interfaces) — modify only if required"
    )
    ignored_files: List[str] = Field(
        default_factory=list,
        description="Files matched by search but irrelevant — do NOT modify or verify"
    )
    blocked_files: List[str] = Field(
        default_factory=list,
        description="Useful reference files that MUST NEVER be modified (generated code, lock files, swagger)"
    )

    # ── Per-file detailed resolutions ──
    file_resolutions: List[FileResolution] = Field(
        default_factory=list,
        description="Per-file edit strategy with reasoning, confidence, and evidence IDs"
    )
    propagation_expectations: List[PropagationExpectation] = Field(
        default_factory=list,
        description="How changes in owner files will propagate to downstream consumers"
    )

    # ── Hypothesis resolution ──
    accepted_hypothesis: str = Field(
        default="",
        description="The hypothesis that best explains the implementation approach"
    )
    rejected_hypotheses: List[str] = Field(
        default_factory=list,
        description="Hypotheses that were considered but rejected"
    )

    # ── Reasoning ──
    edit_strategy_summary: str = Field(
        default="",
        description="1-2 sentence summary of the overall edit strategy"
    )
    owner_reason: str = Field(
        default="",
        description="Why the owner files are the canonical source"
    )
    verification_reason: str = Field(
        default="",
        description="Why verification targets should be verified but not modified"
    )
    supporting_reason: str = Field(
        default="",
        description="Why supporting files may need changes for consistency"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Overall confidence in this decision"
    )
    reasoning: List[str] = Field(
        default_factory=list,
        description="Point-by-point reasoning for the decision"
    )

    # ── Provenance ──
    deterministic_signals: List[DeterministicSignals] = Field(
        default_factory=list,
        description="Raw deterministic signals computed before LLM reasoning"
    )
    evidence_used: List[str] = Field(
        default_factory=list,
        description="File paths of evidence items that informed this decision"
    )


# ============================================================================
# STRUCTURED OUTCOME FINDING MODEL
# ============================================================================

@dataclass
class OutcomeFinding:
    """Canonical structured outcome finding for requirements-verification."""
    verdict: str                  # "CORRECT" | "PARTIAL" | "INCOMPLETE" | "UNCERTAIN"
    requirement_id: str          # e.g., "REQ-1", "REQ-2"
    requirement_text: str        # The requirement being evaluated
    summary: str                 # Human-readable summary of the finding
    offending_code: str          # Exact condition or snippet that failed (e.g. "searchData.length === 1")
    affected_file: str           # Target file path (e.g. "add-members.component.ts")
    affected_line: Optional[int] = None  # Line number if identifiable
    missing_behavior: str = ""   # What must be added to satisfy the ticket
    next_action: str = ""        # Recommended next step (e.g. "Re-evaluating existing evidence and re-planning")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

