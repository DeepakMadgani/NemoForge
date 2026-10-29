"""
CC4E Skill layer for the reasoning kernel.

A Skill is CC4E-specific knowledge for a class of tickets. It does two things:

  1. ADVISES the kernel: which capabilities (tools) to prefer, which validators
     must pass, and which mistakes to avoid.  These are grounded in the real
     CC4E architecture (Spring Boot 3.x / Java 17, Angular 19, Flyway, sagas-lib,
     Spring Cloud Stream + RabbitMQ).
  2. SEEDS deterministic knowledge via prepare(): LLM keyword expansion,
     Brain index lookups, and Feature Graph owner resolution — so the kernel
     starts every ticket from a strong, project-specific position.

The reasoning kernel stays in control.  Skills never execute the loop — they make
the kernel start from a strong, project-specific position (the CC4E advantage).

Design:
  Skill.matches(...)  -> is this skill relevant to the ticket?
  Skill.prepare(ctx)  -> deterministic seeding (returns a short summary string)
  select_skill(...)   -> pick the best-match skill (priority-scored, not first-match)

Playbook definitions below were generated from a deep analysis of the CC4E
codebase (C:/CC4E): project-service, area-service, issues-service,
ene-load-service, sagas-service, bim-gateway, se-connector-apis, xchange-ui.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ticket_to_code.llm_utils import expand_ticket_keywords

logger = logging.getLogger(__name__)

_IGNORE_DIRS = {
    ".git", "node_modules", "dist", "build", "target", "out", "trace",
    "logs", "coverage", ".venv", "venv", "__pycache__",
}


@dataclass
class LiteralInvestigation:
    """Minimal object shaped like the pipeline's investigation result.

    verify_outcome() reads current_state_literals / desired_state_literals, so
    seeding these lets the literal verifier confirm version tickets deterministically.
    """
    current_state_literals: List[str] = field(default_factory=list)
    desired_state_literals: List[str] = field(default_factory=list)
    ticket_type: str = ""


@dataclass
class Skill:
    """A project-specific playbook that intercepts a class of tickets.

    Attributes:
        name: Short, descriptive identifier (e.g. "rest_endpoint").
        matches_keys: Keywords that commonly appear in tickets of this type.
            Used to intercept/route a ticket to this playbook.
        capabilities: Prioritized list of engine tools the agent should use,
            in order.
        validators: Strict conditions that MUST hold before the job is "done".
        guidance: Architectural guardrails grounded in the CC4E codebase.
        common_mistakes: Recurring errors engineers make in CC4E for this task.
        priority: Higher wins when more than one skill matches a ticket.
    """
    name: str
    matches_keys: List[str]
    capabilities: List[str] = field(default_factory=list)
    validators: List[str] = field(default_factory=list)
    guidance: str = ""
    common_mistakes: List[str] = field(default_factory=list)
    priority: int = 100

    def matches(self, ticket_type: str, text: str) -> int:
        """Return the number of matches_keys present in the ticket.

        Returns an integer count (0 = no match).  The caller uses this for
        priority-based selection: the skill with the most keyword hits wins.
        """
        tt = (ticket_type or "").lower()
        tx = (text or "").lower()
        joined = f"{tt} {tx}"
        return sum(1 for key in self.matches_keys if key.lower() in joined)

    def prepare(self, ctx: Any) -> str:
        """Seed the kernel with deterministic knowledge before reasoning starts.

        This base implementation does three things for EVERY skill:
          1. LLM keyword expansion — the LLM autonomously decides the extraction
             level (low/medium/high) based on the ticket context.
          2. Brain index lookup — queries the Brain for owners/patterns/playbooks.
          3. Feature Graph lookup — resolves learned owner files from the Feature
             Graph so the kernel doesn't need to rediscover them.

        Returns a short summary string appended to the scratchpad.
        """
        ticket = ctx.ticket
        text = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}"
        summary_parts: List[str] = []

        # -- 1. LLM keyword expansion --
        try:
            expanded = expand_ticket_keywords(text)
            old_lits = expanded.get("current_state_literals", [])
            new_lits = expanded.get("desired_state_literals", [])

            if old_lits or new_lits:
                existing = getattr(ctx, "investigation", None)
                cur = list(getattr(existing, "current_state_literals", []) or []) if existing else []
                des = list(getattr(existing, "desired_state_literals", []) or []) if existing else []
                merged = LiteralInvestigation(
                    current_state_literals=list(dict.fromkeys(cur + old_lits)),
                    desired_state_literals=list(dict.fromkeys(des + new_lits)),
                    ticket_type=getattr(ctx, "ticket_type", ""),
                )
                setattr(ctx, "investigation", merged)
                summary_parts.append(
                    f"keywords seeded: current={old_lits[:3]}... desired={new_lits[:3]}..."
                )
        except Exception as exc:
            logger.debug("LLM keyword expansion skipped: %s", exc)

        # -- 2. Brain index lookup --
        bm = getattr(ctx, "brain_manager", None)
        if bm is not None:
            try:
                payload = bm.query_ticket(text, getattr(ctx, "ticket_type", ""))
                owners = payload.get("owner_files", [])
                for fp in owners:
                    ctx.candidate_files[fp] = max(ctx.candidate_files.get(fp, 0.0), 0.85)
                if owners:
                    summary_parts.append("brain owners: " + ", ".join(owners[:6]))
                patterns = payload.get("ticket_patterns", [])
                if patterns:
                    summary_parts.append("brain pattern: " + str(patterns[0].get("pattern_id", "")))
                playbooks = payload.get("playbooks", [])
                if playbooks:
                    summary_parts.append("brain playbook: " + str(playbooks[0].get("playbook", "")))
            except Exception as exc:
                summary_parts.append(f"(brain query skipped: {exc})")

        # -- 3. Feature Graph lookup --
        fg = getattr(ctx, "feature_graph", None)
        if fg is not None:
            try:
                ticket_type = getattr(ctx, "ticket_type", "")
                feature = fg.match_feature(text, ticket_type)
                if feature is not None:
                    owners = fg.resolve_owner_candidates(feature, ctx.workspace_path)
                    for o in owners:
                        ctx.candidate_files[o["file"]] = max(
                            ctx.candidate_files.get(o["file"], 0.0), float(o["confidence"])
                        )
                    if owners:
                        learned = [o["file"] for o in owners if o.get("source") == "learned"]
                        hinted = [o["file"] for o in owners if o.get("source") == "hint"]
                        if learned:
                            summary_parts.append("feature-graph LEARNED owners: " + ", ".join(learned[:5]))
                        if hinted:
                            summary_parts.append("feature-graph owner hints (verify before editing): " + ", ".join(hinted[:5]))
                    else:
                        summary_parts.append("feature graph knows the feature but no owners yet -- use find_owner to discover.")
                else:
                    summary_parts.append("no feature match -- discover the owner dynamically (find_owner).")
            except Exception as exc:
                summary_parts.append(f"(feature-graph lookup skipped: {exc})")
        else:
            summary_parts.append("no feature graph available -- discover the owner dynamically (find_owner).")

        if not summary_parts:
            return f"{self.name}: no deterministic seeding; kernel will discover dynamically."
        return f"{self.name} seeded -- " + " | ".join(summary_parts)


# ===========================================================================
# CC4E SKILL PLAYBOOKS
#
# Grounded in the real CC4E architecture:
#   Java services: Spring Boot 3.x / Java 17, mostly Gradle (area-service &
#   se-connector-apis are Maven).
#   Layered: controller -> service(/impl) -> repository(JPA) -> entity/dto,
#   with validators/, sagas/handlers, messaging/, events/, config/, exceptions/.
#   REST: springdoc-openapi (io.swagger.v3) + HATEOAS model assemblers.
#   New code: constructor injection (@RequiredArgsConstructor).
#   Schema: Flyway migrations in src/main/resources/db/migration.
#   Async: Spring Cloud Stream + RabbitMQ (StreamBridge) and sagas-lib.
#   UI: Angular 19 (NgModules + some standalone), selector prefix 'se-',
#   ngx-translate for i18n, RxJS/@ngrx-signals for state.
# ===========================================================================

_REST_ENDPOINT = Skill(
    name="rest_endpoint",
    matches_keys=["endpoint", "rest api", "controller", "swagger", "route", "http", "api_change", "api"],
    capabilities=[
        "brain_query", "recall_feature", "find_owner",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "New handler methods carry a springdoc @Operation (and @ApiResponses) annotation; "
        "the controller keeps its @Tag.",
        "Request bodies use @Valid DTOs (never raw entities); URL paths come from the "
        "constants/URLMappings package, not hardcoded strings.",
        "The owning *-service Gradle/Maven build compiles cleanly and JUnit 5 + Mockito "
        "tests cover the success and failure paths.",
    ],
    guidance=(
        "Endpoints belong in the service's controller package (e.g. "
        "project-service/src/main/java/com/cloudforge/projectservice/controller or "
        "issues-service .../rest/controllers) and MUST delegate to a service interface -- "
        "never call repositories, GraphQL clients, or messaging directly from a controller. "
        "Use constructor injection with @RequiredArgsConstructor + private final fields (do "
        "not extend the legacy @Autowired field-injection pattern) and shape responses with "
        "HATEOAS EntityModel + the existing model assemblers rather than returning JPA entities."
    ),
    common_mistakes=[
        "Returning JPA entities straight from the controller instead of a DTO/HATEOAS model assembler.",
        "Hardcoding URL path strings and HTTP status codes instead of reusing URLMappings/constants.",
        "Adding @Autowired field injection to a new controller instead of constructor injection.",
    ],
    priority=120,
)

_DB_MIGRATION = Skill(
    name="db_migration",
    matches_keys=["migration", "schema", "flyway", "column", "table", "entity", "database"],
    capabilities=[
        "brain_query", "find_owner",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "A new, forward-only Flyway script exists under src/main/resources/db/migration "
        "named V{next-version}__Description.sql; no previously applied migration is edited.",
        "The matching JPA @Entity mapping is updated in lock-step with the migration and "
        "uses explicit @Table/@Column mappings (no reliance on implicit Hibernate naming).",
        "spring.jpa.hibernate.ddl-auto stays 'none' in non-test profiles and the service build passes.",
    ],
    guidance=(
        "Every schema change is a NEW Flyway migration in the service's "
        "src/main/resources/db/migration folder (PostgreSQL) -- never modify an already-applied "
        "V-script. Keep the migration and the JPA entity consistent, and access data through "
        "Spring Data JPA repositories or Specifications/Criteria API. Never concatenate raw SQL/JPQL "
        "with user input."
    ),
    common_mistakes=[
        "Editing an existing applied V-script instead of adding a new higher-versioned migration.",
        "Relying on ddl-auto=update to change the schema instead of writing a Flyway script.",
        "Building JPQL/SQL by string concatenation with user input instead of parameterized "
        "queries or Specifications (SQL-injection risk).",
    ],
    priority=120,
)

_UI_COMPONENT = Skill(
    name="ui_component",
    matches_keys=[
        "component", "angular", "template", "ngx-translate", "ui",
        "css", "style", "layout", "footer", "header", "react", "jsx", "tsx", "frontend",
    ],
    capabilities=[
        "brain_query", "find_owner",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "The component ships the full file set (.ts/.html/.scss/.spec.ts), uses the 'se-' "
        "selector prefix, and is declared/exported by the correct NgModule (or is a standalone "
        "component if the surrounding area already uses loadComponent).",
        "No hardcoded user-facing strings -- all labels/messages use ngx-translate keys under "
        "src/assets/i18n; API access lives in a service (HttpClient/GraphQL), not the component.",
        "Every manual RxJS subscription is cleaned up (currObs[] or takeUntilDestroyed), and "
        "'ng build' / 'ng lint' / 'ng test' pass.",
    ],
    guidance=(
        "UI work lives in xchange-ui/src/app (feature modules) or src/shared (reusable "
        "components). Match the local pattern of the feature you edit -- module-based stays "
        "module-based. Keep components focused on view orchestration and push API/GraphQL/business "
        "logic into the shared services under src/app/modules/shared/services. Surface failures "
        "through NotificationService.notifyError, never console-only, and reuse existing constants."
    ),
    common_mistakes=[
        "Hardcoding display strings instead of adding ngx-translate keys.",
        "Calling HttpClient/GraphQL directly from the component instead of a shared service.",
        "Leaking manual subscriptions by not unsubscribing in ngOnDestroy.",
    ],
    priority=120,
)

_SAGA_HANDLER = Skill(
    name="saga_handler",
    matches_keys=["saga", "handler", "compensate", "workflow", "orchestration"],
    capabilities=[
        "brain_query", "recall_feature", "find_owner",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "The handler extends the correct abstract saga handler and implements process(), "
        "compensate(), backup(), and the required-fields methods.",
        "compensate() is idempotent and safe to call when process() never ran (state flags "
        "are checked first); no mutable request state is stored on the singleton bean.",
        "JUnit 5 + Mockito tests cover both the happy path and the compensation flow.",
    ],
    guidance=(
        "Saga step handlers live in the service's sagas/handlers package and are built on the "
        "sagas-lib. Handlers are Spring singletons -- keep per-request state out of instance fields. "
        "Use constants for saga field names, log entry/exit with the commandId, and make "
        "compensate() fully idempotent."
    ),
    common_mistakes=[
        "Storing mutable request/step state in singleton handler fields (thread-safety bug).",
        "Writing a compensate() that assumes process() completed, so it fails or double-reverts.",
        "Inlining saga field-name strings instead of reusing the constants package.",
    ],
    priority=115,
)

_MESSAGING_EVENT = Skill(
    name="messaging_event",
    matches_keys=["kafka", "rabbitmq", "event", "stream", "producer", "consumer", "message"],
    capabilities=[
        "brain_query", "find_owner",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "Producers use StreamBridge.send() with a binding-name constant, and every new binding "
        "has a matching entry in application.yaml.",
        "Consumers follow the Consumer<Message<T>> pattern and handle deserialization/bad-payload "
        "failures without crashing the listener.",
        "The service build passes and status/state transitions driven by the message stay consistent.",
    ],
    guidance=(
        "Async messaging uses Spring Cloud Stream over RabbitMQ (not raw Kafka clients). Put "
        "producers/consumers in the service's messaging package, drive them with StreamBridge and "
        "binding-name constants, and register any new binding in application.yaml. In pipeline "
        "services like ene-load-service, preserve the Upload->Validate->Process->Report "
        "producer/subscriber pairing."
    ),
    common_mistakes=[
        "Adding a new binding in code but forgetting the corresponding application.yaml entry.",
        "Letting a malformed/undeserializable message crash the consumer instead of handling it.",
        "Hardcoding binding/channel names instead of using the messaging constants.",
    ],
    priority=110,
)

_VALIDATOR = Skill(
    name="validator",
    matches_keys=["validator", "validation", "constraint", "@valid", "field length"],
    capabilities=[
        "brain_query", "recall_feature", "find_owner",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "Validation logic is stateless with no side effects (no service calls or DB writes) and "
        "lives in the validators/ package.",
        "Declarative field validation uses Jakarta Validation annotations (@NotNull/@NotBlank/@Size) "
        "or a proper ConstraintValidator; field limits reuse the shared FieldLengths constants.",
        "JUnit 5 + Mockito/AssertJ tests cover valid and invalid inputs.",
    ],
    guidance=(
        "Custom validators belong in the service's validators/ package and must be reusable and "
        "stateless (extend ConstraintValidator, or the existing SagaRequestValidator base for saga "
        "flows). Never perform service calls or database writes inside a validator, and reuse the "
        "shared FieldLengths / constants rather than inlining magic numbers."
    ),
    common_mistakes=[
        "Putting side effects (service or repository calls) inside validator logic.",
        "Hardcoding field-length limits instead of using the shared FieldLengths constants.",
        "Duplicating validation already available via Jakarta Validation annotations.",
    ],
    priority=110,
)

_VERSION_CHANGE = Skill(
    name="version_change",
    matches_keys=["version_bump", "version", "release", "bump"],
    capabilities=[
        "brain_query", "brain_search_feature", "brain_get_owner_files",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "Old version literal no longer present in canonical/config files.",
        "New version literal present in the canonical source.",
        "Build succeeds after the version change.",
    ],
    guidance=(
        "Version tickets change a canonical version literal. Prefer the seeded canonical "
        "owner (application.yml/properties, pom.xml, build.gradle, package.json). Change the "
        "source of truth, ensure consumers derive from it, then verify the old literal is gone "
        "and the new one is present. Do NOT edit UI display text directly."
    ),
    common_mistakes=[
        "Editing UI/display files instead of the canonical config owner.",
        "Leaving the old version string in some files.",
        "Search-and-replace without confirming the canonical source.",
    ],
    priority=120,
)

_CONFIGURATION = Skill(
    name="configuration",
    matches_keys=["configuration", "config", "dependency", "deployment", "yaml", "properties"],
    capabilities=[
        "brain_query", "brain_search_feature", "brain_get_owner_files",
        "grep", "read_file", "write_patch", "verify_outcome",
    ],
    validators=[
        "Only the owning config file(s) changed.",
        "Exact config key updated.",
        "No code changes when only config needed to change.",
    ],
    guidance=(
        "Configuration tickets modify config files (yml/properties/json/xml/env). Confirm the "
        "exact key and change only the owning config file(s). In CC4E, application.yml is under "
        "src/main/resources/ in each service. Do not change Java code when only a config value "
        "needs updating."
    ),
    common_mistakes=[
        "Changing code when only config needed to change.",
        "Editing the wrong service's config file.",
        "Not checking all profiles (dev/staging/prod) for the config key.",
    ],
    priority=110,
)

_ACCESS_CONTROL = Skill(
    name="access_control",
    matches_keys=[
        "access_control", "authorization", "auth", "permission", "rbac",
        "role", "coordinator", "co-ordinator", "organization", "organisation",
        "private issues", "forbidden", "read only",
    ],
    capabilities=[
        "brain_query", "brain_get_playbook", "brain_get_root_causes", "find_owner",
        "search_code", "grep", "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "Role/org restriction enforced in owner policy/service layer.",
        "Default/explicit exception behavior preserved.",
        "Unauthorized cross-organization access blocked.",
    ],
    guidance=(
        "Access-control tickets are policy tickets first, UI tickets second. Identify the "
        "authoritative authorization/validation owner, implement minimal policy checks there, "
        "then verify edit forms and APIs reflect the restriction without bypass paths. In CC4E, "
        "auth flows through ot2-auth and role checks live in service-layer validators."
    ),
    common_mistakes=[
        "Only disabling UI control while leaving backend authorization open.",
        "Applying restriction globally without preserving defined exceptions.",
        "Not testing cross-organization access scenarios.",
    ],
    priority=120,
)

_BUGFIX = Skill(
    name="bugfix",
    matches_keys=["bug", "fix", "npe", "error", "exception", "regression", "defect", "crash", "race", "performance"],
    capabilities=[
        "recall_memory", "brain_query", "brain_get_root_causes",
        "find_owner", "read_file", "grep",
        "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "The fix targets the root-cause file/path only; no opportunistic unrelated refactors.",
        "Exceptions are handled explicitly (log-and-rethrow or map to the bim-commons-lib "
        "exception hierarchy) -- no empty catch blocks, System.out.println, or printStackTrace.",
        "A regression test reproduces the bug and passes after the fix; the owning build is green.",
    ],
    guidance=(
        "Constrain edits to the root-cause path and keep a clear symptom->cause->fix chain. "
        "Route errors through the existing exception hierarchy and @ControllerAdvice handlers "
        "(GenericControllerExceptionHandler), use parameterized @Slf4j logging, and never leak "
        "stack traces, tokens, or DB constraint names to callers."
    ),
    common_mistakes=[
        "Swallowing exceptions with an empty catch or logging without rethrowing/handling.",
        "Fixing the symptom in a consumer file while the real owner/root-cause file is untouched.",
        "Bundling unrelated cleanup into the bugfix, expanding the blast radius.",
    ],
    priority=90,
)

# Catch-all so the engine can still guide *any* ticket with CC4E guardrails.
_CC4E_DEFAULT = Skill(
    name="cc4e_default",
    matches_keys=[],
    capabilities=[
        "recall_memory", "brain_query", "find_owner",
        "read_file", "write_patch", "run_build", "verify_outcome",
    ],
    validators=[
        "Changes are minimal and scoped to the owner file(s) discovered for the ticket.",
        "The owning service build passes (Gradle for most services, Maven for area-service / "
        "se-connector-apis; 'ng build' for xchange-ui).",
        "No plaintext secrets, no System.out.println/console-only logging, no swallowed exceptions.",
    ],
    guidance=(
        "Respect the CC4E layered architecture (controller -> service -> repository -> entity for "
        "Java services; feature-module components + shared services for xchange-ui). Prefer additive, "
        "backward-compatible changes, reuse existing services/constants/shared libraries "
        "(bim-commons-lib, sagas-lib, ot2-auth), and keep the diff as small as the ticket allows."
    ),
    common_mistakes=[
        "Editing consumer files when the real change belongs in an owner file.",
        "Broad refactors or new abstractions the ticket did not ask for.",
        "Bypassing existing shared services/constants and reimplementing them.",
    ],
    priority=0,
)


# -- Skill registry (highest priority first, DEFAULT last as catch-all) --

_SKILLS: List[Skill] = [
    _REST_ENDPOINT,
    _DB_MIGRATION,
    _UI_COMPONENT,
    _VERSION_CHANGE,
    _ACCESS_CONTROL,
    _CONFIGURATION,
    _SAGA_HANDLER,
    _MESSAGING_EVENT,
    _VALIDATOR,
    _BUGFIX,
]


def select_skill(ticket_type: str, ticket_text: str) -> Skill:
    """Pick the most relevant CC4E skill for this ticket.

    Uses priority-based scoring: the skill with the most keyword hits wins.
    Ties are broken by priority.  Falls back to _CC4E_DEFAULT when nothing
    matches.
    """
    best = _CC4E_DEFAULT
    best_hits = 0
    for skill in _SKILLS:
        hits = skill.matches(ticket_type, ticket_text)
        if hits > best_hits or (hits == best_hits and hits > 0 and skill.priority > best.priority):
            best, best_hits = skill, hits
    return best
