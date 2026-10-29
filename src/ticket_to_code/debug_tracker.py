"""
WORKFLOW DEBUG TRACKER - Single file capturing ALL outputs from each phase
Usage: Run workflow, all outputs saved to debug_workflow_output.log
"""

import json
import os
from datetime import datetime
from pathlib import Path

class WorkflowDebugger:
    def __init__(self, output_file='debug_workflow_output.log'):
        self.output_file = output_file
        self.logs = []
        self._init_file()
    
    def _init_file(self):
        """Initialize output file with header"""
        with open(self.output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write(f"AVIATOR WORKFLOW DEBUG LOG - {datetime.now()}\n")
            f.write("="*100 + "\n\n")
    
    def log_phase(self, phase_name, data=None, llm_input=None, llm_output=None, agent_name=None):
        """Log a workflow phase with all details"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase_name,
            'agent': agent_name,
            'llm_input': llm_input,
            'llm_output': llm_output,
            'data': data
        }
        
        with open(self.output_file, 'a') as f:
            f.write(f"\n[{entry['timestamp']}] PHASE: {phase_name}\n")
            if agent_name:
                f.write(f"  AGENT: {agent_name}\n")
            
            if llm_input:
                f.write(f"\n  LLM INPUT:\n{self._format_json(llm_input)}\n")
            
            if llm_output:
                f.write(f"\n  LLM OUTPUT:\n{self._format_json(llm_output)}\n")
            
            if data:
                f.write(f"\n  DATA:\n{self._format_json(data)}\n")
            
            f.write("-"*100 + "\n")
        
        self.logs.append(entry)
    
    def _format_json(self, data):
        """Format data for readable output"""
        try:
            if isinstance(data, str):
                return data[:500]  # Truncate long strings
            return json.dumps(data, indent=2)[:500]
        except:
            return str(data)[:500]
    
    def log_error(self, phase, error, context=None):
        """Log errors"""
        with open(self.output_file, 'a') as f:
            f.write(f"\n[ERROR] {phase}\n")
            f.write(f"  Error: {str(error)}\n")
            if context:
                f.write(f"  Context: {context}\n")
            f.write("-"*100 + "\n")
    
    def summary(self):
        """Print summary of all phases"""
        with open(self.output_file, 'a') as f:
            f.write("\n\n" + "="*100 + "\n")
            f.write("WORKFLOW SUMMARY\n")
            f.write("="*100 + "\n")
            
            phases_executed = [log['phase'] for log in self.logs]
            f.write(f"\nPhases Executed: {len(phases_executed)}\n")
            for phase in phases_executed:
                f.write(f"  ✓ {phase}\n")
            
            agents_used = set(log['agent'] for log in self.logs if log['agent'])
            f.write(f"\nAgents Used: {len(agents_used)}\n")
            for agent in agents_used:
                f.write(f"  ✓ {agent}\n")

# Global debugger instance
_debugger = None

def get_debugger():
    global _debugger
    if _debugger is None:
        _debugger = WorkflowDebugger()
    return _debugger

def log_phase(phase, data=None, llm_input=None, llm_output=None, agent=None):
    """Convenience function to log from anywhere in code"""
    get_debugger().log_phase(phase, data, llm_input, llm_output, agent)

def log_error(phase, error, context=None):
    """Convenience function to log errors"""
    get_debugger().log_error(phase, error, context)

def print_summary():
    """Print workflow summary"""
    get_debugger().summary()
