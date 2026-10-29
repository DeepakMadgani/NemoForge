"""
Ticket Analyzer Agent

Converts unstructured ValueEdge tickets into structured, machine-readable requirements.

This is Agent 1 in the autonomous pipeline.

Author: Deepak Madgani
Date: April 2026
"""

import json
import logging
import re
from typing import Optional, List

from ticket_to_code.llm_utils import llm_invoke

from ticket_to_code.runtime.llm_factory import LLMRegistry
from langchain_core.messages import SystemMessage, HumanMessage

from ticket_to_code.models import (
    ValueEdgeTicket, 
    StructuredRequirements,
    SolutionGuidance,
    InvestigationResult
)
from ticket_to_code.json_utils import parse_llm_json

logger = logging.getLogger(__name__)


class TicketAnalyzerAgent:
    """
    Agent 1: Ticket Understanding & Requirement Extraction
    
    Converts natural language user stories into structured requirements
    using LLM-powered analysis.
    """
    
    def __init__(self):
        """Initialize with Aviator LLM"""
        self.llm = LLMRegistry.get_llm()
        logger.info("Ticket Analyzer Agent initialized")
    
    def analyze_ticket(self, ticket: ValueEdgeTicket) -> StructuredRequirements:
        """
        Analyze ticket and extract structured requirements.
        
        Args:
            ticket: ValueEdge ticket to analyze
            
        Returns:
            Structured requirements with functional, technical, edge cases, etc.
            
        Raises:
            RuntimeError: If analysis fails
        """
        logger.info(f"Analyzing ticket: {ticket.ticket_id}")
        
        try:
            # Build analysis prompt
            system_prompt = self._build_analysis_prompt()
            user_prompt = self._build_user_prompt(ticket)
            
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt
            
            # Invoke LLM
            response = llm_invoke(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Parse response
            try:
                requirements = self._parse_response(response.content)
            except ValueError as parse_error:
                logger.warning(
                    "Falling back to deterministic ticket requirements due to invalid LLM JSON: %s",
                    parse_error,
                )
                requirements = self._build_fallback_requirements(ticket, response.content, parse_error)
            
            logger.info(f"Analysis complete: {len(requirements.functional_requirements)} functional reqs")
            return requirements
            
        except Exception as e:
            logger.error(f"Ticket analysis failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to analyze ticket {ticket.ticket_id}: {e}")
    
    def _build_analysis_prompt(self) -> str:
        """Build system prompt for requirement analysis"""
        return """You are an expert software requirements analyst with deep knowledge of:
- Software development lifecycle (SDLC)
- User story analysis
- Acceptance criteria decomposition
- Edge case identification
- Test-driven development

YOUR TASK:
Analyze the provided user story/ticket and extract structured requirements.

EXTRACT:
1. Functional Requirements - What the system should DO
2. Technical Requirements - HOW to implement (specific code changes, APIs, components)
3. Edge Cases - Error scenarios, boundary conditions, unexpected inputs
4. Acceptance Tests - Test cases to verify functionality
5. Affected Components - Which modules/files will change  5. Affected Components - Which modules/files will change
  6. Non-Functional Requirements - Performance, security, scalability needs
  7. Risk - (low, medium, high)
  8. Complexity - (low, medium, high)
  9. Ticket Category - (Feature, Bug, Config, Refactor)
  10. Expected Files - Specific file paths expected
  11. Estimated Work - (small, medium, large)


RULES:
- Be SPECIFIC - avoid vague requirements like "improve performance"
- Include file paths if mentioned (e.g., "Update src/components/CheckoutForm.tsx")
- Extract ALL edge cases mentioned or implied
- Generate testable acceptance criteria
- Identify ALL affected components

RESPOND with JSON ONLY using this exact structure:
{
  "functional_requirements": ["Req 1", "Req 2"],
  "technical_requirements": ["Tech req 1", "Tech req 2"],
  "edge_cases": ["Edge case 1", "Edge case 2"],
  "acceptance_tests": ["Test 1", "Test 2"],
  "affected_components": ["Component 1", "Component 2"],
  "non_functional_requirements": ["NFR 1", "NFR 2"]
}

NO markdown, NO explanations, ONLY JSON."""
    
    def _build_user_prompt(self, ticket: ValueEdgeTicket) -> str:
        """Build user prompt with ticket details"""
        return f"""
TICKET ID: {ticket.ticket_id}
TITLE: {ticket.title}
PRIORITY: {ticket.priority.value}

DESCRIPTION:
{ticket.description}

ACCEPTANCE CRITERIA:
{self._format_acceptance_criteria(ticket.acceptance_criteria)}

LABELS: {', '.join(ticket.labels) if ticket.labels else 'None'}

Now extract structured requirements in JSON format.
        """.strip()
    
    def _format_acceptance_criteria(self, criteria: list[str]) -> str:
        """Format acceptance criteria for prompt"""
        if not criteria:
            return "None specified"
        return "\n".join(f"- {c}" for c in criteria)
    
    def _parse_response(self, response_content: str) -> StructuredRequirements:
        """
        Parse LLM response into StructuredRequirements.
        
        Args:
            response_content: Raw LLM response
            
        Returns:
            Validated StructuredRequirements object
        """
        try:
            data = parse_llm_json(response_content, expect_list=False)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response in ticket analyzer")
            raise ValueError(f"Invalid JSON response from LLM: {e}")

        required_fields = [
            "functional_requirements",
            "technical_requirements",
            "edge_cases",
            "acceptance_tests",
            "affected_components",
        ]
        if not isinstance(data, dict):
            raise ValueError("LLM requirement payload must be a JSON object")
        for field in required_fields:
            if field not in data:
                data[field] = []
        
        # Validate with Pydantic
        try:
            return StructuredRequirements.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid requirement payload: {e}")

    def _build_fallback_requirements(
        self,
        ticket: ValueEdgeTicket,
        raw_response: str,
        parse_error: Exception,
    ) -> StructuredRequirements:
        """Deterministic fallback when requirement-analysis JSON is malformed."""
        acceptance = [str(x).strip() for x in (ticket.acceptance_criteria or []) if str(x).strip()]
        if not acceptance:
            acceptance = ["Implementation satisfies ticket intent without unrelated changes."]

        labels = [str(x).strip() for x in (ticket.labels or []) if str(x).strip()]
        preview = (raw_response or "").strip()[:300]

        payload = {
            "functional_requirements": acceptance,
            "technical_requirements": [
                "Apply minimal scoped code changes tied directly to ticket intent.",
                "Preserve existing behavior outside requested scope.",
            ],
            "edge_cases": [
                "Malformed model JSON output during analysis should not fail workflow.",
            ],
            "acceptance_tests": acceptance,
            "affected_components": [ticket.title.strip() or ticket.ticket_id],
            "non_functional_requirements": [
                f"Robustness fallback triggered due to parse error: {parse_error}",
                "No unrelated file modifications.",
            ],
        }

        # Include available label hints in technical requirements for better planning context.
        if labels:
            payload["technical_requirements"].append("Ticket labels: " + ", ".join(labels))
        if preview:
            payload["technical_requirements"].append("Model response preview used for fallback context.")

        return StructuredRequirements.model_validate(payload)
    
    def generate_solution_guidance(
        self, 
        ticket: ValueEdgeTicket,
        investigation: InvestigationResult,
        rag_context: Optional[List[dict]] = None
    ) -> SolutionGuidance:
        """
        Generate solution guidance for non-code issues (configuration, usage, etc.).
        
        Args:
            ticket: ValueEdge ticket
            investigation: Investigation result with root cause and recommendations
            rag_context: Retrieved documentation, config examples, troubleshooting guides
            
        Returns:
            Detailed solution guidance with step-by-step instructions
        """
        logger.info(f"Generating solution guidance for: {ticket.ticket_id}")
        logger.info(f"  Using {len(rag_context) if rag_context else 0} context documents from RAG")
        
        try:
            system_prompt = self._build_solution_prompt()
            user_prompt = self._build_solution_user_prompt(ticket, investigation, rag_context)
            
            # Invoke LLM
            response = llm_invoke(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Parse response
            try:
                result = self._parse_solution_response(response.content)
            except ValueError as parse_error:
                # Do not fail the whole workflow on malformed JSON from the LLM.
                logger.warning(
                    "Falling back to deterministic solution guidance due to invalid LLM JSON: %s",
                    parse_error,
                )
                result = self._build_fallback_solution_guidance(
                    ticket=ticket,
                    investigation=investigation,
                    raw_response=response.content,
                    parse_error=parse_error,
                )
            
            logger.info(f"Solution guidance generated: {result.solution_type}")
            return result
            
        except Exception as e:
            logger.error(f"Solution guidance generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate solution guidance: {e}")
    
    def _build_solution_prompt(self) -> str:
        """Build system prompt for solution guidance"""
        return """You are a senior technical support engineer and system administrator.

YOUR ROLE:
Provide clear, actionable solutions for configuration, usage, and deployment issues.

YOUR TASK:
Analyze the ticket and investigation findings, then provide:
1. Clear root cause explanation
2. Step-by-step solution instructions
3. Configuration examples (with actual code/config snippets)
4. Verification steps
5. Common mistakes to avoid

SOLUTION TYPES:
- configuration: Config file or environment setup issue
- usage: Incorrect usage or API call
- deployment: Deployment or infrastructure issue
- authentication: Auth/credential problem
- networking: Network connectivity issue
- permissions: Access control or permission issue

GUIDELINES:
✅ Be specific - provide exact commands, config values, file paths
✅ Include examples - show actual configuration snippets
✅ Be practical - focus on what the user needs to DO
✅ Think like support - anticipate common mistakes
✅ Provide verification - how to confirm it's fixed

RESPONSE FORMAT (JSON only):
{
  "issue_summary": "User cannot login to CloudForge",
  "root_cause": "Missing or incorrect API_Type parameter in CoreContent configuration",
  "solution_type": "configuration",
  "step_by_step_solution": [
    "Open the CoreContent configuration file at: C:\\\\Program Files\\\\CloudForge\\\\CloudForge\\\\config\\\\corecontent.xml",
    "Locate the <Authentication> section",
    "Add or update the API_Type parameter: <API_Type>OTDS</API_Type>",
    "Save the configuration file",
    "Restart the CloudForge service",
    "Test the login again"
  ],
  "configuration_examples": [
    "<Authentication>\\n  <API_Type>OTDS</API_Type>\\n  <ServiceUrl>https://otds-server:8443/otdsws</ServiceUrl>\\n  <ResourceID>cloudforge</ResourceID>\\n</Authentication>"
  ],
  "verification_steps": [
    "Open CloudForge web interface",
    "Enter valid credentials",
    "Verify successful login and dashboard access",
    "Check logs at: C:\\\\Program Files\\\\CloudForge\\\\CloudForge\\\\logs\\\\auth.log for 'Authentication successful'"
  ],
  "common_mistakes": [
    "Using wrong API_Type value (OTCS instead of OTDS)",
    "Incorrect ServiceUrl format (missing /otdsws path)",
    "Firewall blocking port 8443",
    "Invalid ResourceID not matching OTDS configuration"
  ],
  "related_documentation": [
    "CloudForge Authentication Guide: https://docs.cloudforge.dev/auth",
    "OTDS Integration: https://docs.cloudforge.dev/integration"
  ],
  "escalation_criteria": "If configuration is correct and issue persists, check: 1) OTDS service health, 2) Network connectivity to OTDS server, 3) Valid OTDS resource registration. Escalate to infrastructure team if all checks pass."
}

NO markdown, NO explanations outside JSON, ONLY JSON."""
    
    def _build_solution_user_prompt(
        self,
        ticket: ValueEdgeTicket,
        investigation: InvestigationResult,
        rag_context: Optional[List[dict]] = None
    ) -> str:
        """Build user prompt for solution guidance with RAG context"""
        
        # Format RAG context if available
        context_section = ""
        if rag_context:
            context_section = "\n\nRELEVANT DOCUMENTATION AND CONFIGURATION EXAMPLES:\n"
            for idx, chunk in enumerate(rag_context[:10], 1):  # Limit to 10 chunks
                context_section += f"\n--- Example {idx} ---\n"
                context_section += f"Source: {chunk.get('file_path', 'Unknown')}\n"
                context_section += f"Content:\n{chunk.get('content', '')[:500]}\n"  # Limit length
        
        return f"""
TICKET ID: {ticket.ticket_id}
TITLE: {ticket.title}

DESCRIPTION:
{ticket.description}

INVESTIGATION FINDINGS:
- Ticket Type: {investigation.ticket_type.value}
- Root Cause Hypothesis: {investigation.root_cause_hypothesis or 'Not specified'}
- Recommended Action: {investigation.recommended_action}
- Possible Causes: {', '.join(investigation.possible_causes)}
- Investigation Areas: {', '.join(investigation.investigation_areas)}
- Affected Systems: {', '.join(investigation.affected_systems)}
{context_section}

Using the investigation findings AND the retrieved documentation/configuration examples above,
provide detailed step-by-step solution guidance. Use ACTUAL examples from the context!
        """.strip()
    
    def _parse_solution_response(self, response_content: str) -> SolutionGuidance:
        """Parse LLM response into SolutionGuidance"""
        try:
            data = parse_llm_json(response_content, expect_list=False)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse solution JSON")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        
        return SolutionGuidance.model_validate(data)

    def _extract_first_json_object(self, text: str) -> Optional[dict]:
        """Best-effort extraction for malformed responses that still contain a JSON object."""
        if not text:
            return None

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        raw = match.group(0)
        candidates = [raw, re.sub(r",\s*([}\]])", r"\1", raw)]
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        return None

    def _build_fallback_solution_guidance(
        self,
        ticket: ValueEdgeTicket,
        investigation: InvestigationResult,
        raw_response: str,
        parse_error: Exception,
    ) -> SolutionGuidance:
        """Return deterministic guidance when LLM JSON is malformed."""
        summary = ticket.title.strip() or f"Guidance for {ticket.ticket_id}"
        root_cause = (
            (investigation.root_cause_hypothesis or "").strip()
            or "Model returned malformed structured output for guidance generation."
        )
        solution_type = (getattr(investigation.ticket_type, "value", None) or "configuration").strip()
        recommended_action = str(investigation.recommended_action or "").strip()

        steps: List[str] = []
        if recommended_action:
            steps.append(recommended_action)
        steps.append("Apply the recommendation in the impacted config/deployment surface identified by investigation.")
        steps.append("Re-run the failing workflow or command and confirm the original error no longer appears.")

        verification_steps: List[str] = [
            "Validate that the original ticket symptom is resolved in runtime behavior.",
            "Check service logs for absence of the prior error signature.",
        ]

        common_mistakes = [c for c in investigation.possible_causes if c][:4]
        response_preview = (raw_response or "").strip()[:400]
        configuration_examples = [response_preview] if response_preview else []

        escalation = (
            "Escalate if the issue persists after applying the recommendation and validating logs. "
            f"Fallback reason: {parse_error}"
        )

        return SolutionGuidance(
            issue_summary=summary,
            root_cause=root_cause,
            solution_type=solution_type,
            step_by_step_solution=steps,
            configuration_examples=configuration_examples,
            verification_steps=verification_steps,
            common_mistakes=common_mistakes,
            related_documentation=[],
            escalation_criteria=escalation,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def analyze_ticket(ticket: ValueEdgeTicket) -> StructuredRequirements:
    """
    Convenience function to analyze a ticket.
    
    Args:
        ticket: ValueEdge ticket
        
    Returns:
        Structured requirements
    """
    agent = TicketAnalyzerAgent()
    return agent.analyze_ticket(ticket)
