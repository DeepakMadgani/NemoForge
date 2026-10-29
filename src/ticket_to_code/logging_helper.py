"""
Logging Helper for Autonomous Ticket-to-Code System

Creates detailed, user-friendly log files that explain what the system did
and why it made certain decisions.

Author: Deepak Madgani
Date: April 2026
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from ticket_to_code.models import (
    ValueEdgeTicket,
    InvestigationResult,
    AutonomousWorkflowState,
    WorkflowStatus,
    TicketType
)


class WorkflowLogger:
    """Creates beautiful, user-readable log files for workflow execution"""
    
    def __init__(self, logs_dir: Path = None):
        """Initialize logger with log directory"""
        if logs_dir is None:
            logs_dir = Path(__file__).parent.parent.parent / "logs"
        
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.logs_dir / "investigations").mkdir(exist_ok=True)
        (self.logs_dir / "code_generation").mkdir(exist_ok=True)
        (self.logs_dir / "errors").mkdir(exist_ok=True)
        
    def create_investigation_log(
        self,
        ticket: ValueEdgeTicket,
        investigation_result: InvestigationResult,
        project_path: str
    ) -> Path:
        """Create a detailed investigation report log file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticket.ticket_id}_{timestamp}_investigation.log"
        log_path = self.logs_dir / "investigations" / filename
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("AUTONOMOUS TICKET-TO-CODE SYSTEM - INVESTIGATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Ticket Information
            f.write("TICKET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Ticket ID:      {ticket.ticket_id}\n")
            f.write(f"Title:          {ticket.title}\n")
            f.write(f"Priority:       {ticket.priority.value}\n")
            f.write(f"Labels:         {', '.join(ticket.labels)}\n")
            f.write(f"Submitted:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nDescription:\n{ticket.description}\n\n")
            
            if ticket.acceptance_criteria:
                f.write("Acceptance Criteria:\n")
                for i, criteria in enumerate(ticket.acceptance_criteria, 1):
                    f.write(f"  {i}. {criteria}\n")
                f.write("\n")
            
            # Project Information
            f.write("-" * 80 + "\n")
            f.write(f"Project Path:   {project_path}\n")
            f.write(f"Analysis Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Investigation Results
            f.write("=" * 80 + "\n")
            f.write("INVESTIGATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Ticket Type:           {investigation_result.ticket_type.value.upper()}\n")
            f.write(f"Code Changes Needed:   {'YES' if investigation_result.requires_code_changes else 'NO'}\n")
            f.write(f"Confidence Level:      {investigation_result.confidence * 100:.1f}%\n")
            f.write(f"Needs More Context:    {'YES' if investigation_result.needs_more_context else 'NO'}\n")
            f.write("\n")
            
            # Root Cause (if investigation/bug)
            if investigation_result.root_cause_hypothesis:
                f.write("ROOT CAUSE HYPOTHESIS\n")
                f.write("-" * 80 + "\n")
                f.write(f"{investigation_result.root_cause_hypothesis}\n\n")
            
            # Affected Systems
            if investigation_result.affected_systems:
                f.write("AFFECTED SYSTEMS\n")
                f.write("-" * 80 + "\n")
                for system in investigation_result.affected_systems:
                    f.write(f"  • {system}\n")
                f.write("\n")
            
            # Investigation Areas
            if investigation_result.investigation_areas:
                f.write("INVESTIGATION AREAS\n")
                f.write("-" * 80 + "\n")
                for area in investigation_result.investigation_areas:
                    f.write(f"  • {area}\n")
                f.write("\n")
            
            # Possible Causes
            if investigation_result.possible_causes:
                f.write("POSSIBLE CAUSES\n")
                f.write("-" * 80 + "\n")
                for i, cause in enumerate(investigation_result.possible_causes, 1):
                    f.write(f"  {i}. {cause}\n")
                f.write("\n")
            
            # Recommended Action
            f.write("RECOMMENDED ACTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{investigation_result.recommended_action}\n\n")
            
            # Detailed Explanation
            f.write("DETAILED EXPLANATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{investigation_result.explanation}\n\n")
            
            # Decision
            f.write("=" * 80 + "\n")
            f.write("WORKFLOW DECISION\n")
            f.write("=" * 80 + "\n\n")
            
            if not investigation_result.requires_code_changes:
                f.write("DECISION: NO CODE GENERATION REQUIRED\n\n")
                f.write("The following phases will be SKIPPED:\n")
                f.write("  ❌ Phase 1 - Requirements Analysis\n")
                f.write("  ❌ Phase 2 - Architecture Planning\n")
                f.write("  ❌ Phase 3 - RAG Context Retrieval\n")
                f.write("  ❌ Phase 4 - Code Generation\n")
                f.write("  ❌ Phase 5 - Build Execution\n")
                f.write("  ❌ Phase 6 - Test Execution\n")
                f.write("  ❌ Phase 7 - Debug & Fix\n")
                f.write("  ❌ Phase 8 - Integration\n\n")
                
                f.write("OUTCOME: Investigation report provided to user.\n")
                f.write("STATUS: explanation_provided\n\n")
            else:
                f.write("DECISION: CODE GENERATION REQUIRED\n\n")
                f.write("The following phases will be EXECUTED:\n")
                f.write("  ✅ Phase 1 - Requirements Analysis\n")
                f.write("  ✅ Phase 2 - Architecture Planning\n")
                f.write("  ✅ Phase 3 - RAG Context Retrieval\n")
                f.write("  ✅ Phase 4 - Code Generation\n")
                f.write("  ✅ Phase 5 - Build Execution\n")
                f.write("  ✅ Phase 6 - Test Execution\n")
                f.write("  ✅ Phase 7 - Debug & Fix (if needed)\n")
                f.write("  ✅ Phase 8 - Integration\n\n")
                
                f.write("Proceeding to requirements analysis phase...\n\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF INVESTIGATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log File: {log_path}\n")
        
        return log_path
    
    def append_solution_guidance_to_log(
        self,
        log_path: Path,
        solution_guidance
    ) -> None:
        """
        Append RAG-based solution guidance to existing investigation log.
        
        This adds the DETAILED solution (with RAG context) after the initial 
        investigation triage.
        
        Args:
            log_path: Path to existing investigation log file
            solution_guidance: SolutionGuidance object with RAG-based solution
        """
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("DETAILED SOLUTION GUIDANCE (WITH RAG CONTEXT)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ISSUE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"{solution_guidance.issue_summary}\n\n")
            
            f.write("ROOT CAUSE (VERIFIED WITH CODEBASE ANALYSIS)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{solution_guidance.root_cause}\n\n")
            
            f.write("SOLUTION TYPE\n")
            f.write("-" * 80 + "\n")
            f.write(f"{solution_guidance.solution_type.upper()}\n\n")
            
            # Step-by-step solution
            f.write("STEP-BY-STEP SOLUTION\n")
            f.write("-" * 80 + "\n")
            for i, step in enumerate(solution_guidance.step_by_step_solution, 1):
                f.write(f"{i}. {step}\n")
            f.write("\n")
            
            # Configuration examples (from RAG!)
            if solution_guidance.configuration_examples:
                f.write("CONFIGURATION EXAMPLES (FROM CODEBASE)\n")
                f.write("-" * 80 + "\n")
                for i, example in enumerate(solution_guidance.configuration_examples, 1):
                    f.write(f"\nExample {i}:\n")
                    f.write("```\n")
                    f.write(f"{example}\n")
                    f.write("```\n")
                f.write("\n")
            
            # Verification steps
            if solution_guidance.verification_steps:
                f.write("VERIFICATION STEPS\n")
                f.write("-" * 80 + "\n")
                for i, step in enumerate(solution_guidance.verification_steps, 1):
                    f.write(f"{i}. {step}\n")
                f.write("\n")
            
            # Common mistakes
            if solution_guidance.common_mistakes:
                f.write("COMMON MISTAKES TO AVOID\n")
                f.write("-" * 80 + "\n")
                for i, mistake in enumerate(solution_guidance.common_mistakes, 1):
                    f.write(f"{i}. {mistake}\n")
                f.write("\n")
            
            # Related documentation
            if solution_guidance.related_documentation:
                f.write("RELATED DOCUMENTATION\n")
                f.write("-" * 80 + "\n")
                for doc in solution_guidance.related_documentation:
                    f.write(f"  • {doc}\n")
                f.write("\n")
            
            # Escalation criteria
            if solution_guidance.escalation_criteria:
                f.write("ESCALATION CRITERIA\n")
                f.write("-" * 80 + "\n")
                f.write(f"{solution_guidance.escalation_criteria}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("NOTE: Above solution generated using RAG-retrieved codebase examples\n")
            f.write("=" * 80 + "\n")
    
    def create_workflow_log(
        self,
        ticket: ValueEdgeTicket,
        final_state: AutonomousWorkflowState,
        project_path: str
    ) -> Path:
        """Create complete workflow execution log"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticket.ticket_id}_{timestamp}_workflow.log"
        log_path = self.logs_dir / "code_generation" / filename
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("AUTONOMOUS TICKET-TO-CODE SYSTEM - COMPLETE WORKFLOW LOG\n")
            f.write("=" * 80 + "\n\n")
            
            # Ticket Info
            f.write("TICKET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Ticket ID:      {ticket.ticket_id}\n")
            f.write(f"Title:          {ticket.title}\n")
            f.write(f"Priority:       {ticket.priority.value}\n")
            f.write(f"Project Path:   {project_path}\n\n")
            
            # Final Status
            f.write("EXECUTION SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Final Status:   {final_state.status.value}\n")
            f.write(f"Duration:       {final_state.end_time - final_state.start_time if final_state.end_time else 'In Progress'}\n")
            f.write(f"\nPhases Executed:\n")
            
            phase_names = {
                "investigation": "Phase 0 - Investigation",
                "analysis": "Phase 1 - Requirements Analysis",
                "planning": "Phase 2 - Architecture Planning",
                "rag": "Phase 3 - RAG Context Retrieval",
                "generation": "Phase 4 - Code Generation",
                "build": "Phase 5 - Build Execution",
                "test": "Phase 6 - Test Execution",
                "debug": "Phase 7 - Debug & Fix",
                "integration": "Phase 8 - Integration"
            }
            
            for phase_key, phase_name in phase_names.items():
                phase_data = getattr(final_state, phase_key, None)
                if phase_data:
                    f.write(f"  ✅ {phase_name}\n")
                else:
                    f.write(f"  ⏭️  {phase_name} (skipped)\n")
            
            f.write("\n")
            
            # Generated Files
            if final_state.generated_code:
                f.write("GENERATED FILES\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Files Generated: {len(final_state.generated_code)}\n\n")
                for i, file_info in enumerate(final_state.generated_code, 1):
                    f.write(f"{i}. {file_info.relative_path}\n")
                    f.write(f"   Type: {file_info.file_type}\n")
                    f.write(f"   Lines: {len(file_info.content.splitlines())}\n\n")
            else:
                f.write("NO CODE FILES GENERATED\n")
                f.write("-" * 80 + "\n")
                f.write("This ticket did not require code generation.\n\n")
            
            # Errors (if any)
            if final_state.errors:
                f.write("ERRORS ENCOUNTERED\n")
                f.write("-" * 80 + "\n")
                for error in final_state.errors:
                    f.write(f"  • {error}\n")
                f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF WORKFLOW LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nLog Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log File: {log_path}\n")
        
        return log_path
    
    def create_summary_log(
        self,
        ticket: ValueEdgeTicket,
        investigation_result: Optional[InvestigationResult],
        final_state: Optional[AutonomousWorkflowState],
        project_path: str
    ) -> Path:
        """Create a quick summary log (1-page overview)"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticket.ticket_id}_{timestamp}_summary.txt"
        log_path = self.logs_dir / filename
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("AUTONOMOUS TICKET-TO-CODE SYSTEM\n")
            f.write("Quick Summary Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Ticket:  {ticket.ticket_id} - {ticket.title}\n")
            f.write(f"Project: {project_path}\n")
            f.write(f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if investigation_result:
                f.write(f"Type:    {investigation_result.ticket_type.value.upper()}\n")
                f.write(f"Action:  {'Code Generation' if investigation_result.requires_code_changes else 'Investigation Only'}\n")
                f.write(f"Status:  {'explanation_provided' if not investigation_result.requires_code_changes else 'processing'}\n\n")
            
            if final_state:
                f.write(f"Final Status: {final_state.status.value}\n")
                f.write(f"Files Generated: {len(final_state.generated_files) if final_state.generated_files else 0}\n\n")
            
            f.write(f"\nFor detailed report, see:\n")
            f.write(f"  - logs/investigations/{ticket.ticket_id}_{timestamp}_investigation.log\n")
            if final_state and final_state.generated_code:
                f.write(f"  - logs/code_generation/{ticket.ticket_id}_{timestamp}_workflow.log\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        return log_path
    
    def log_console_output(self, log_path: Path):
        """Print log file location to console"""
        print("\n" + "=" * 80)
        print("📄 DETAILED REPORT SAVED")
        print("=" * 80)
        print(f"\n✅ Log file created: {log_path}")
        print(f"\nYou can review the complete analysis at:")
        print(f"  {log_path.absolute()}\n")
        print("=" * 80 + "\n")


# Convenience function
def create_investigation_log(
    ticket: ValueEdgeTicket,
    investigation_result: InvestigationResult,
    project_path: str,
    logs_dir: Optional[Path] = None
) -> Path:
    """Quick function to create investigation log"""
    logger = WorkflowLogger(logs_dir)
    log_path = logger.create_investigation_log(ticket, investigation_result, project_path)
    logger.log_console_output(log_path)
    return log_path
