"""
Evidence Graph - Track reasoning chain for debugging and transparency

Built continuously during discovery and planning phases.
Provides full traceability: "Why did we choose these methods?"
Answer: "Search result X → Method Y → API Z → Requirement W → Ticket"

Author: Deepak Madgani
Date: April 2026
"""

import logging
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes in evidence graph"""
    SEARCH_RESULT = "search_result"    # From RAG search
    METHOD = "method"                  # Code method
    API = "api_call"                   # API or function call
    REQUIREMENT = "requirement"        # From ticket requirements
    TICKET = "ticket"                  # The original ticket
    DEPENDENCY = "dependency"          # Code dependency
    EDGE_WEIGHT = "edge_weight"        # Weighted connection
    CONFIDENCE = "confidence"          # Confidence assessment


class RelationType(str, Enum):
    """Types of relationships between nodes"""
    IMPLEMENTS = "implements"          # Method implements requirement
    USES = "uses"                      # Method uses another method
    SATISFIES = "satisfies"           # Method satisfies requirement
    DEPENDS_ON = "depends_on"          # Depends on another method
    RELATES_TO = "relates_to"          # General relation
    CONTRADICTS = "contradicts"        # Conflicts with


@dataclass
class EvidenceNode:
    """Node in evidence graph"""
    id: str
    node_type: NodeType
    content: Dict                       # Content varies by type
    confidence: float                   # 0-1 confidence in this node
    created_at: datetime = None
    source: str = "unknown"             # Where did this come from
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class EvidenceEdge:
    """Edge connecting two nodes"""
    from_node_id: str
    to_node_id: str
    relation_type: RelationType
    evidence: str                       # Why we believe this connection
    weight: float = 1.0                 # Weight of connection (0-1)
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class EvidenceGraph:
    """
    Track evidence chain for full transparency.
    
    Allows debugging:
    - "Why did you pick this method?"
    - "What evidence supports the root cause?"
    - "How confident are you in the solution?"
    - "What alternatives did you consider?"
    """
    
    def __init__(self, ticket_id: str = "unknown"):
        """Initialize evidence graph"""
        self.ticket_id = ticket_id
        self.nodes: Dict[str, EvidenceNode] = {}
        self.edges: List[EvidenceEdge] = []
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"[EVIDENCE] Initializing graph for ticket {ticket_id}")
    
    def add_search_result(
        self,
        query: str,
        file_path: str,
        method_name: str,
        code_snippet: str,
        relevance_score: float
    ) -> str:
        """
        Add search result node (from discovery phase).
        
        Returns:
            Node ID for later reference
        """
        
        node_id = f"search_{len(self.nodes)}"
        
        node = EvidenceNode(
            id=node_id,
            node_type=NodeType.SEARCH_RESULT,
            content={
                'query': query,
                'file': file_path,
                'method': method_name,
                'snippet': code_snippet
            },
            confidence=relevance_score,
            source='rag_search'
        )
        
        self.nodes[node_id] = node
        self.logger.debug(f"[EVIDENCE] Added search result: {node_id} (confidence: {relevance_score:.2f})")
        
        return node_id
    
    def add_method_node(
        self,
        file_path: str,
        class_name: str,
        method_name: str,
        method_signature: str,
        confidence: float = 1.0
    ) -> str:
        """
        Add method node (from localization phase).
        
        Returns:
            Node ID for later reference
        """
        
        node_id = f"method_{file_path}_{class_name}_{method_name}".replace(' ', '_')
        
        node = EvidenceNode(
            id=node_id,
            node_type=NodeType.METHOD,
            content={
                'file': file_path,
                'class': class_name,
                'method': method_name,
                'signature': method_signature
            },
            confidence=confidence,
            source='localization'
        )
        
        self.nodes[node_id] = node
        self.logger.debug(f"[EVIDENCE] Added method node: {node_id}")
        
        return node_id
    
    def add_api_node(
        self,
        api_name: str,
        module: str,
        purpose: str,
        confidence: float = 1.0
    ) -> str:
        """
        Add API/function call node.
        
        Returns:
            Node ID for later reference
        """
        
        node_id = f"api_{module}_{api_name}".replace(' ', '_')
        
        node = EvidenceNode(
            id=node_id,
            node_type=NodeType.API,
            content={
                'name': api_name,
                'module': module,
                'purpose': purpose
            },
            confidence=confidence,
            source='code_analysis'
        )
        
        self.nodes[node_id] = node
        self.logger.debug(f"[EVIDENCE] Added API node: {node_id}")
        
        return node_id
    
    def add_requirement_node(
        self,
        requirement_text: str,
        requirement_type: str,
        confidence: float = 1.0
    ) -> str:
        """
        Add requirement node (from ticket).
        
        Returns:
            Node ID for later reference
        """
        
        node_id = f"req_{len([n for n in self.nodes.values() if n.node_type == NodeType.REQUIREMENT])}"
        
        node = EvidenceNode(
            id=node_id,
            node_type=NodeType.REQUIREMENT,
            content={
                'text': requirement_text,
                'type': requirement_type
            },
            confidence=confidence,
            source='ticket_analysis'
        )
        
        self.nodes[node_id] = node
        self.logger.debug(f"[EVIDENCE] Added requirement node: {node_id}")
        
        return node_id
    
    def add_ticket_node(
        self,
        ticket_id: str,
        title: str,
        description: str
    ) -> str:
        """
        Add ticket node (root of evidence graph).
        
        Returns:
            Node ID for later reference
        """
        
        node_id = f"ticket_{ticket_id}"
        
        node = EvidenceNode(
            id=node_id,
            node_type=NodeType.TICKET,
            content={
                'id': ticket_id,
                'title': title,
                'description': description
            },
            confidence=1.0,
            source='user_input'
        )
        
        self.nodes[node_id] = node
        self.logger.debug(f"[EVIDENCE] Added ticket node: {node_id}")
        
        return node_id
    
    def connect(
        self,
        from_node_id: str,
        to_node_id: str,
        relation_type: RelationType,
        evidence: str,
        weight: float = 1.0
    ):
        """
        Connect two nodes with evidence.
        
        Args:
            from_node_id: Source node
            to_node_id: Target node
            relation_type: Type of relationship
            evidence: Why this connection exists
            weight: Strength of connection (0-1)
        """
        
        if from_node_id not in self.nodes:
            self.logger.warning(f"[EVIDENCE] Source node {from_node_id} not found")
            return
        
        if to_node_id not in self.nodes:
            self.logger.warning(f"[EVIDENCE] Target node {to_node_id} not found")
            return
        
        edge = EvidenceEdge(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relation_type=relation_type,
            evidence=evidence,
            weight=weight
        )
        
        self.edges.append(edge)
        self.logger.debug(f"[EVIDENCE] Connected {from_node_id} → {to_node_id} ({relation_type.value})")
    
    def get_reasoning_chain(self, method_id: str) -> List[Dict]:
        """
        Get full reasoning chain for why a method was chosen.
        
        Returns reasoning path:
        Ticket → Requirement → Search Result → Method
        
        Returns:
            List of nodes in reasoning path
        """
        
        chain = []
        
        # Find all paths leading to this method
        visited = set()
        
        def trace_back(node_id: str, depth: int = 0):
            if node_id in visited or depth > 10:
                return
            
            visited.add(node_id)
            
            if node_id in self.nodes:
                node = self.nodes[node_id]
                chain.append({
                    'depth': depth,
                    'node_id': node_id,
                    'type': node.node_type.value,
                    'confidence': node.confidence,
                    'content': node.content
                })
            
            # Find incoming edges
            for edge in self.edges:
                if edge.to_node_id == node_id:
                    trace_back(edge.from_node_id, depth + 1)
        
        trace_back(method_id)
        
        # Reverse to get ticket → method order
        chain.reverse()
        
        self.logger.info(f"[EVIDENCE] Reasoning chain for {method_id}: {len(chain)} steps")
        
        return chain
    
    def get_confidence_chain(self, method_id: str) -> float:
        """
        Calculate overall confidence for choosing a method.
        
        Confidence = average confidence of all nodes in reasoning chain.
        
        Returns:
            Confidence score (0-1)
        """
        
        chain = self.get_reasoning_chain(method_id)
        
        if not chain:
            return 0.0
        
        confidences = [node['confidence'] for node in chain]
        avg_confidence = sum(confidences) / len(confidences)
        
        self.logger.info(f"[EVIDENCE] Overall confidence for {method_id}: {avg_confidence:.2f}")
        
        return avg_confidence
    
    def visualize(self) -> str:
        """
        Generate Graphviz DOT format for visualization.
        
        Can be visualized with: graphviz, Graphviz Online, etc.
        
        Returns:
            DOT format string
        """
        
        lines = ["digraph evidence_graph {"]
        lines.append(f'  label="Evidence Graph for {self.ticket_id}";')
        lines.append("  rankdir=TB;")
        
        # Add nodes
        for node_id, node in self.nodes.items():
            label = f"{node.node_type.value}\\n{node.confidence:.2f}"
            color = self._get_node_color(node.node_type)
            lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}", style=filled];')
        
        # Add edges
        for edge in self.edges:
            label = edge.relation_type.value
            lines.append(
                f'  "{edge.from_node_id}" → "{edge.to_node_id}" '
                f'[label="{label}", weight={edge.weight}];'
            )
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _get_node_color(self, node_type: NodeType) -> str:
        """Get color for node type in visualization"""
        
        colors = {
            NodeType.TICKET: "lightblue",
            NodeType.REQUIREMENT: "lightgreen",
            NodeType.SEARCH_RESULT: "lightyellow",
            NodeType.METHOD: "lightcoral",
            NodeType.API: "plum",
            NodeType.DEPENDENCY: "khaki",
        }
        
        return colors.get(node_type, "lightgray")
    
    def export_json(self) -> Dict:
        """
        Export graph as JSON for storage/debugging.
        
        Returns:
            {
                'ticket_id': str,
                'created_at': datetime,
                'nodes': List[Dict],
                'edges': List[Dict]
            }
        """
        
        return {
            'ticket_id': self.ticket_id,
            'created_at': datetime.now().isoformat(),
            'nodes': [asdict(node) for node in self.nodes.values()],
            'edges': [asdict(edge) for edge in self.edges]
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the graph"""
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': {
                node_type.value: len([n for n in self.nodes.values() if n.node_type == node_type])
                for node_type in NodeType
            },
            'avg_confidence': (
                sum(n.confidence for n in self.nodes.values()) / len(self.nodes)
                if self.nodes else 0.0
            ),
            'avg_node_degree': (
                2 * len(self.edges) / len(self.nodes)
                if self.nodes else 0.0
            )
        }
