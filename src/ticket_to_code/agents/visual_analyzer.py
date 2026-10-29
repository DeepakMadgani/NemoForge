"""
Visual / Screenshot Analysis Agent — Enhancement 8

Multimodal LLM analysis of attached screenshots:
- Extract error text from UI screenshots
- Identify UI state and status indicators
- Feed visual analysis into investigation result

Safety: READ-ONLY. Only reads image files.

Author: Deepak Madgani
Date: July 2026
"""

import base64
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class VisualElement(BaseModel):
    """An element detected in a screenshot."""
    element_type: str = Field(..., description="error_dialog | status_bar | log_output | ui_state | text")
    content: str = Field("", description="Text or description of the element")
    location: str = Field("", description="Where in the image (top, center, bottom)")
    severity: str = Field("info", description="info | warning | error")


class ImageAnalysisResult(BaseModel):
    """Analysis result for an individual ordered screenshot."""
    image_index: int = Field(1, description="1-based index of the image (1, 2, 3...)")
    image_label: str = Field("Image 1", description="Formatted label (e.g. Image 1, Image 2)")
    image_name: str = Field("", description="Filename of the image")
    image_path: str = Field("", description="Absolute path to the image file")
    elements: List[VisualElement] = Field(default_factory=list)
    error_text: Optional[str] = Field(None, description="Extracted error text from this image")
    ui_state: Optional[str] = Field(None, description="Description of UI state for this image")
    ui_literals: List[str] = Field(default_factory=list, description="Exact visible text strings (headers, breadcrumbs, buttons)")
    layout_issues: List[str] = Field(default_factory=list, description="Layout defects detected (e.g. overflow behind footer)")
    diagnosis: Optional[str] = Field(None, description="LLM analysis of this screenshot")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class VisualAnalysis(BaseModel):
    """Result of visual screenshot analysis across all attached images."""
    has_visual_data: bool = Field(False)
    image_path: Optional[str] = Field(None)
    images: List[ImageAnalysisResult] = Field(default_factory=list)
    elements: List[VisualElement] = Field(default_factory=list)
    error_text: Optional[str] = Field(None, description="Combined/primary error text")
    ui_state: Optional[str] = Field(None, description="Combined/primary UI state description")
    ui_literals: List[str] = Field(default_factory=list, description="Extracted UI literals across all images")
    layout_issues: List[str] = Field(default_factory=list, description="Extracted layout issues across all images")
    diagnosis: Optional[str] = Field(None, description="Combined LLM synthesis across screenshots")
    formatted_summary: Optional[str] = Field(None, description="Markdown formatted summary of all visual evidence")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# Supported image formats
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


class VisualAnalyzer:
    """
    Analyzes screenshots attached to tickets using multimodal LLM.
    Supports sequential multi-image analysis (Image 1, Image 2, etc.) for before/after and navigation workflows.
    """

    def __init__(self, workspace_path: str = ""):
        self.workspace = Path(workspace_path) if workspace_path else None

    def _find_screenshots(self, attachment_paths: List[str]) -> List[Path]:
        """Find valid screenshot files from attachment paths preserving user order."""
        valid = []
        for p in attachment_paths:
            path = Path(p)
            if not path.is_absolute() and self.workspace and (self.workspace / path).exists():
                path = self.workspace / path
            if path.exists() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                if path.stat().st_size < 15_000_000:  # Max 15MB
                    valid.append(path)
        return valid

    def _encode_image(self, path: Path) -> str:
        """Encode image to base64 for multimodal LLM."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        attachment_paths: List[str],
        ticket_text: str = "",
    ) -> VisualAnalysis:
        """
        Analyze all screenshot attachments sequentially using multimodal LLM.

        Args:
            attachment_paths: Paths to attached image files.
            ticket_text: Ticket description for context.

        Returns:
            VisualAnalysis with ordered findings per image and synthesis.
        """
        screenshots = self._find_screenshots(attachment_paths)
        if not screenshots:
            logger.info("VisualAnalyzer: no valid screenshots found")
            return VisualAnalysis(has_visual_data=False)

        logger.info(f"VisualAnalyzer: analyzing {len(screenshots)} screenshot(s) in sequential order")
        image_results: List[ImageAnalysisResult] = []
        all_elements: List[VisualElement] = []
        errors = []
        ui_states = []
        diagnoses = []

        for idx, screenshot_path in enumerate(screenshots, start=1):
            label = f"Image {idx}"
            logger.info(f"VisualAnalyzer: analyzing [{label}] {screenshot_path.name}")
            try:
                img_res = self._llm_analyze_single(screenshot_path, idx, label, ticket_text, total_images=len(screenshots))
                image_results.append(img_res)
                all_elements.extend(img_res.elements)
                if img_res.error_text:
                    errors.append(f"[{label}] {img_res.error_text}")
                if img_res.ui_state:
                    ui_states.append(f"[{label}] {img_res.ui_state}")
                if img_res.diagnosis:
                    diagnoses.append(f"[{label}] {img_res.diagnosis}")
            except Exception as exc:
                logger.warning(f"VisualAnalyzer: analysis failed for [{label}] {screenshot_path.name}: {exc}")
                image_results.append(
                    ImageAnalysisResult(
                        image_index=idx,
                        image_label=label,
                        image_name=screenshot_path.name,
                        image_path=str(screenshot_path),
                        diagnosis=f"Analysis failed for {label}: {exc}",
                        confidence=0.0,
                    )
                )

        combined_error = " | ".join(errors) if errors else None
        combined_ui_state = " | ".join(ui_states) if ui_states else None
        combined_diagnosis = "\n".join(diagnoses) if diagnoses else None

        all_literals: List[str] = []
        all_layout_issues: List[str] = []
        for res in image_results:
            for lit in res.ui_literals:
                if lit and lit not in all_literals:
                    all_literals.append(lit)
            for iss in res.layout_issues:
                if iss and iss not in all_layout_issues:
                    all_layout_issues.append(iss)

        # Build clean markdown summary with clear numbered badges
        summary_lines = []
        for res in image_results:
            line = f"- **{res.image_label}** (`{res.image_name}`):"
            details = []
            if res.error_text:
                details.append(f"UI Error: '{res.error_text}'")
            if res.ui_state:
                details.append(f"State: {res.ui_state}")
            if res.layout_issues:
                details.append(f"Layout Issues: {', '.join(res.layout_issues)}")
            if res.ui_literals:
                details.append(f"UI Literals: {', '.join(res.ui_literals[:5])}")
            if res.diagnosis:
                details.append(f"Diagnosis: {res.diagnosis}")
            if details:
                line += " " + " | ".join(details)
            summary_lines.append(line)

        formatted_summary = "\n".join(summary_lines)

        return VisualAnalysis(
            has_visual_data=True,
            image_path=str(screenshots[0]) if screenshots else None,
            images=image_results,
            elements=all_elements,
            error_text=combined_error,
            ui_state=combined_ui_state,
            ui_literals=all_literals,
            layout_issues=all_layout_issues,
            diagnosis=combined_diagnosis,
            formatted_summary=formatted_summary,
            confidence=0.9 if image_results else 0.0,
        )

    def _llm_analyze_single(
        self, 
        image_path: Path, 
        index: int, 
        label: str, 
        ticket_text: str,
        total_images: int = 1
    ) -> ImageAnalysisResult:
        """Analyze a single screenshot using multimodal LLM with order context."""
        from ticket_to_code.llm_utils import llm_invoke
        from ticket_to_code.runtime.llm_factory import LLMRegistry
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        image_b64 = self._encode_image(image_path)
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

        prompt = f"""Analyze screenshot {label} (of {total_images} total attachments) from a software application.

Ticket context: {ticket_text[:600]}

Identify:
1. Any error messages or dialogs visible in {label}
2. The current UI state (what screen/page/component is shown)
3. Exact visible UI literal text strings: table column headers (e.g. 'Project Name', 'Total contracts', 'Project code'), breadcrumbs (e.g. 'All projects'), button labels, or visible row keys/values for codebase localization.
4. Any layout or visual defects: e.g. table rows overflowing or hidden behind sticky footer/pagination bar, missing scrollbar, clipped content, overlapping elements.
5. Any status indicators (loading, success, failure, badge, dropdown)
6. Any log output, stack trace, or console messages visible

Respond with JSON:
{{
  "elements": [
    {{
      "element_type": "error_dialog | status_bar | log_output | ui_state | text",
      "content": "the text or description",
      "location": "top | center | bottom | left | right",
      "severity": "info | warning | error"
    }}
  ],
  "error_text": "any error text extracted from this image (null if none)",
  "ui_state": "brief description of the UI state in this image",
  "ui_literals": ["list of exact text strings visible in UI, e.g. 'Total contracts', 'Project code', 'All projects'"],
  "layout_issues": ["list of layout defects visible, e.g. 'Table rows overflow behind sticky footer pagination bar'"],
  "diagnosis": "what this screenshot specifically tells us about the problem or navigation step",
  "confidence": 0.0-1.0
}}"""

        messages = [
            SystemMessage(content=f"You are a UI analysis expert. Analyze {label} to extract error information, UI literals, layout issues, and visual context."),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_b64}",
                    },
                },
            ]),
        ]

        llm = LLMRegistry.get_llm(assistant=False)
        result = llm_invoke(llm, messages)

        if not result or not result.content:
            return ImageAnalysisResult(
                image_index=index,
                image_label=label,
                image_name=image_path.name,
                image_path=str(image_path),
                confidence=0.0,
            )

        raw = result.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ImageAnalysisResult(
                image_index=index,
                image_label=label,
                image_name=image_path.name,
                image_path=str(image_path),
                diagnosis=result.content[:500],
                confidence=0.3,
            )

        elements = [
            VisualElement(
                element_type=e.get("element_type", "text"),
                content=e.get("content", ""),
                location=e.get("location", ""),
                severity=e.get("severity", "info"),
            )
            for e in data.get("elements", [])
        ]

        ui_literals = [str(x).strip() for x in data.get("ui_literals", []) if str(x).strip()]
        layout_issues = [str(x).strip() for x in data.get("layout_issues", []) if str(x).strip()]

        return ImageAnalysisResult(
            image_index=index,
            image_label=label,
            image_name=image_path.name,
            image_path=str(image_path),
            elements=elements,
            error_text=data.get("error_text"),
            ui_state=data.get("ui_state"),
            ui_literals=ui_literals,
            layout_issues=layout_issues,
            diagnosis=data.get("diagnosis"),
            confidence=float(data.get("confidence", 0.8)),
        )
