"""
Visual interpretation tools using VLM (Local/Cloud).

Provides visual understanding of page images retrieved by visual search.
"""

import base64
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Tuple, TYPE_CHECKING

from api.services.tools.base import Tool
from api.schemas.agent import Result, Error, Status

if TYPE_CHECKING:
    from api.services.environment import TreeData

logger = logging.getLogger(__name__)


class VisualInterpretationTool(Tool):
    """
    Tool that interprets visual content using a VLM (Qwen3-VL-8B local, Gemini cloud).
    
    Only available when ColQwen search results exist in the environment.
    Takes page images and generates descriptions of diagrams, charts, etc.
    """
    
    def __init__(self):
        super().__init__(
            name="visual_interpretation",
            description=(
                "Analyze and interpret visual content from retrieved page images. "
                "Use this after colqwen_search to understand diagrams, schematics, "
                "charts, wiring layouts, or any visual elements on the pages. "
                "The VLM will describe what it sees and answer questions about the visuals."
            ),
            status="Analyzing visual content...",
            inputs={
                "prompt": {
                    "description": "Question or instruction about the visual content",
                    "type": "str",
                    "default": "Describe the key visual elements on this page, including any diagrams, charts, or schematics.",
                },
                "page_indices": {
                    "description": "Which pages to analyze (indices into ColQwen results). Omit to analyze all.",
                    "type": "list",
                    "default": None,
                },
                "max_pages": {
                    "description": "Maximum number of pages to analyze",
                    "type": "int",
                    "default": 2,
                },
            },
            end=False,
        )
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """Only available when ColQwen results exist in environment."""
        colqwen_data = tree_data.environment.find("colqwen_search")
        if not colqwen_data:
            return False
        
        # Check if there are actual results
        for name, entries in colqwen_data.items():
            for entry in entries:
                if entry.get("objects"):
                    return True
        return False
    
    async def run_if_true(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Auto-trigger for queries that explicitly ask about visual interpretation.
        """
        query_lower = tree_data.user_prompt.lower()
        
        # Strong visual interpretation indicators
        interpret_keywords = [
            "explain the diagram",
            "describe the schematic",
            "what does the figure show",
            "interpret the chart",
            "analyze the image",
            "what's in the picture",
        ]
        
        # Only auto-trigger if we have ColQwen results AND query asks for interpretation
        if any(kw in query_lower for kw in interpret_keywords):
            if await self.is_tool_available(tree_data):
                return True, {
                    "prompt": f"Based on the user's question: '{tree_data.user_prompt}', describe what you see in this technical document page.",
                }
        
        return False, {}
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """
        Interpret visual content from ColQwen search results.
        
        Gets page images from environment and sends them to VLM.
        """
        from api.core.providers import get_vlm, get_visual_search
        from api.core.config import get_settings
        
        prompt = inputs.get(
            "prompt",
            "Describe the key visual elements on this page, including any diagrams, charts, or schematics."
        )
        page_indices = inputs.get("page_indices")
        max_pages = inputs.get("max_pages", 2)
        
        yield Status("Gathering page images from visual search results...")
        
        # Get ColQwen results from environment
        colqwen_data = tree_data.environment.find("colqwen_search")
        if not colqwen_data:
            yield Error(
                message="No visual search results found in environment",
                recoverable=True,
                suggestion="Run colqwen_search first to retrieve visual content",
            )
            return
        
        # Collect page images
        pages_to_analyze = []
        for name, entries in colqwen_data.items():
            for entry in entries:
                for obj in entry.get("objects", []):
                    pages_to_analyze.append({
                        "page_id": obj.get("page_id"),
                        "page_number": obj.get("page_number"),
                        "asset_manual": obj.get("asset_manual"),
                        "image_path": obj.get("image_path"),
                        "score": obj.get("maxsim_score") if obj.get("maxsim_score") is not None else obj.get("score", 0),
                    })
        
        if not pages_to_analyze:
            yield Error(
                message="No pages found in visual search results",
                recoverable=True,
            )
            return
        
        # Filter by indices if specified
        if page_indices is not None:
            pages_to_analyze = [
                pages_to_analyze[i] for i in page_indices
                if i < len(pages_to_analyze)
            ]
        
        # Limit number of pages
        pages_to_analyze = pages_to_analyze[:max_pages]
        
        yield Status(f"Analyzing {len(pages_to_analyze)} page(s) with VLM...")
        
        # Get VLM provider
        vlm = get_vlm()
        visual_search = get_visual_search()
        settings = get_settings()
        
        # Check if VLM is available
        if not await vlm.is_available():
            yield Error(
                message="VLM service is not available",
                recoverable=True,
                suggestion="Ensure local MLX server is running or cloud API keys are set",
            )
            return
        
        # Process each page
        interpretations = []
        for page_info in pages_to_analyze:
            page_num = page_info["page_number"]
            manual = page_info.get("asset_manual", "Unknown")
            page_id = page_info.get("page_id")
            
            image_data = None  # Will be path or base64 string
            
            # 1. Try local file path first (Local Mode)
            if page_info.get("image_path"):
                # Resolve relative path
                full_path = Path(page_info["image_path"])
                if not full_path.is_absolute():
                    # Assume relative to project root
                    project_root = Path(__file__).parent.parent.parent.parent
                    full_path = project_root / page_info["image_path"]
                
                if full_path.exists():
                    image_data = str(full_path)
            
            # 2. Try fetching from provider (Cloud Mode)
            if not image_data and page_id is not None:
                try:
                    image_bytes = await visual_search.get_page_image(page_id)
                    if image_bytes:
                        # Convert to base64 for VLM
                        image_data = base64.b64encode(image_bytes).decode("utf-8")
                except Exception as e:
                    logger.error(f"Failed to fetch image for page {page_num}: {e}")
            
            if not image_data:
                logger.warning(f"Could not load image for page {page_num}")
                interpretations.append({
                    "page_number": page_num,
                    "asset_manual": manual,
                    "error": "Image content unavailable",
                })
                continue
            
            try:
                yield Status(f"Interpreting page {page_num}...")
                
                # Build a context-aware prompt
                full_prompt = (
                    f"This is page {page_num} from a technical manual called '{manual}'. "
                    f"{prompt}"
                )
                
                interpretation = await vlm.interpret_image(
                    image_path=image_data,
                    prompt=full_prompt,
                    max_tokens=512,
                )
                
                interpretations.append({
                    "page_number": page_num,
                    "asset_manual": manual,
                    "interpretation": interpretation,
                    "score": page_info.get("score"),
                })
                
            except Exception as e:
                logger.error(f"VLM interpretation failed for page {page_num}: {e}")
                interpretations.append({
                    "page_number": page_num,
                    "asset_manual": manual,
                    "error": str(e),
                })
        
        # Check if we got any successful interpretations
        successful = [i for i in interpretations if "interpretation" in i]
        
        if not successful:
            yield Error(
                message="Failed to interpret any pages",
                recoverable=True,
                error_type="vlm_error",
            )
            return
        
        # Build LLM message summarizing what was found
        summary_parts = []
        for interp in successful:
            page = interp["page_number"]
            summary_parts.append(f"Page {page}: {interp['interpretation'][:200]}...")
        
        llm_message = (
            f"Visual interpretation of {len(successful)} page(s):\n" +
            "\n".join(summary_parts)
        )
        
        yield Result(
            objects=interpretations,
            metadata={
                "prompt": prompt,
                "pages_analyzed": len(pages_to_analyze),
                "successful": len(successful),
                "failed": len(interpretations) - len(successful),
            },
            name="visual_interpretations",
            llm_message=llm_message,
        )


class DiagramExtractionTool(Tool):
    """
    Specialized tool for extracting structured information from diagrams.
    
    Focuses on wiring diagrams, circuit schematics, and technical drawings.
    """
    
    def __init__(self):
        super().__init__(
            name="diagram_extraction",
            description=(
                "Extract structured information from technical diagrams. "
                "Best for wiring diagrams, circuit schematics, and block diagrams. "
                "Returns components, connections, and annotations found in the diagram."
            ),
            status="Extracting diagram information...",
            inputs={
                "diagram_type": {
                    "description": "Type of diagram (wiring, circuit, block, flowchart)",
                    "type": "str",
                    "default": "auto",
                },
                "extract_connections": {
                    "description": "Whether to extract component connections",
                    "type": "bool",
                    "default": True,
                },
            },
            end=False,
        )
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """Available when visual interpretation results exist."""
        visual_data = tree_data.environment.find("visual_interpretation")
        return visual_data is not None
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Extract structured information from diagrams."""
        from api.core.providers import get_vlm, get_visual_search
        
        diagram_type = inputs.get("diagram_type", "auto")
        
        yield Status("Analyzing diagram structure...")
        
        # Get visual interpretation results to know what we have
        visual_data = tree_data.environment.find("visual_interpretation")
        if not visual_data:
            yield Error(
                message="No visual interpretation results found",
                recoverable=True,
                suggestion="Run visual_interpretation first",
            )
            return
        
        # Get page images from ColQwen results
        colqwen_data = tree_data.environment.find("colqwen_search")
        if not colqwen_data:
            yield Error(
                message="No search results to extract diagrams from",
                recoverable=True,
            )
            return
        
        # Build extraction prompt based on diagram type
        if diagram_type == "wiring":
            prompt = (
                "Analyze this wiring diagram and extract:\n"
                "1. All component labels and their types\n"
                "2. Wire connections (from -> to)\n"
                "3. Color codes if visible\n"
                "4. Terminal designations\n"
                "Format as structured data."
            )
        elif diagram_type == "circuit":
            prompt = (
                "Analyze this circuit schematic and extract:\n"
                "1. Component types and values (resistors, capacitors, etc.)\n"
                "2. Node connections\n"
                "3. Input/output terminals\n"
                "4. Power supply connections\n"
                "Format as structured data."
            )
        else:
            prompt = (
                "Analyze this technical diagram and extract:\n"
                "1. All labeled components or elements\n"
                "2. Connections or relationships between elements\n"
                "3. Any annotations, values, or specifications shown\n"
                "Format as structured data."
            )
        
        vlm = get_vlm()
        visual_search = get_visual_search()
        
        if not await vlm.is_available():
            yield Error(
                message="VLM service is not available",
                recoverable=True,
            )
            return
        
        # Process first page with diagram content
        extractions = []
        for name, entries in colqwen_data.items():
            for entry in entries:
                for obj in entry.get("objects", [])[:1]:  # Just first page
                    page_id = obj.get("page_id")
                    page_num = obj.get("page_number")
                    image_data = None
                    
                    # Local
                    if obj.get("image_path"):
                        full_path = Path(obj["image_path"])
                        if not full_path.is_absolute():
                            full_path = Path(__file__).parent.parent.parent.parent / obj["image_path"]
                        if full_path.exists():
                            image_data = str(full_path)
                    
                    # Cloud
                    if not image_data and page_id is not None:
                        try:
                            image_bytes = await visual_search.get_page_image(page_id)
                            if image_bytes:
                                image_data = base64.b64encode(image_bytes).decode("utf-8")
                        except Exception as e:
                            logger.error(f"Failed to get image: {e}")
                    
                    if not image_data:
                        continue
                    
                    try:
                        extraction = await vlm.interpret_image(
                            image_path=image_data,
                            prompt=prompt,
                            max_tokens=1024,
                        )
                        
                        extractions.append({
                            "page_number": page_num,
                            "diagram_type": diagram_type,
                            "extraction": extraction,
                        })
                        
                    except Exception as e:
                        logger.error(f"Diagram extraction failed: {e}")
        
        if not extractions:
            yield Error(
                message="Failed to extract diagram information",
                recoverable=True,
            )
            return
        
        yield Result(
            objects=extractions,
            metadata={
                "diagram_type": diagram_type,
                "count": len(extractions),
            },
            name="diagram_extractions",
            llm_message=f"Extracted structured information from {len(extractions)} diagram(s).",
        )
