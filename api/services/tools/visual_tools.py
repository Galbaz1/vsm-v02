"""
Visual interpretation tools using MLX VLM.

Provides visual understanding of page images retrieved by ColQwen search.
"""

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
    Tool that interprets visual content using Qwen3-VL-8B via MLX.
    
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
        
        Gets page images from environment and sends them to MLX VLM.
        """
        from api.services.llm import get_mlx_vlm_client
        
        prompt = inputs.get(
            "prompt",
            "Describe the key visual elements on this page, including any diagrams, charts, or schematics."
        )
        page_indices = inputs.get("page_indices")
        max_pages = inputs.get("max_pages", 2)
        
        yield Status("Gathering page images from ColQwen results...")
        
        # Get ColQwen results from environment
        colqwen_data = tree_data.environment.find("colqwen_search")
        if not colqwen_data:
            yield Error(
                message="No ColQwen search results found in environment",
                recoverable=True,
                suggestion="Run colqwen_search first to retrieve visual content",
            )
            return
        
        # Collect page images
        pages_to_analyze = []
        for name, entries in colqwen_data.items():
            for entry in entries:
                for obj in entry.get("objects", []):
                    image_path = obj.get("image_path")
                    if image_path:
                        pages_to_analyze.append({
                            "page_number": obj.get("page_number"),
                            "asset_manual": obj.get("asset_manual"),
                            "image_path": image_path,
                            "score": obj.get("maxsim_score", 0),
                        })
        
        if not pages_to_analyze:
            yield Error(
                message="No page images found in ColQwen results",
                recoverable=True,
                suggestion="ColQwen results may not include image paths",
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
        
        # Get VLM client
        vlm_client = get_mlx_vlm_client()
        
        # Check if VLM is available
        is_available = await vlm_client.is_available()
        if not is_available:
            yield Error(
                message="MLX VLM server is not available",
                recoverable=True,
                suggestion="Start the VLM server with: mlx_vlm.server --model mlx-community/Qwen3-VL-8B-Instruct-4bit --port 8000",
            )
            return
        
        # Process each page
        interpretations = []
        for page_info in pages_to_analyze:
            image_path = page_info["image_path"]
            page_num = page_info["page_number"]
            manual = page_info.get("asset_manual", "Unknown")
            
            # Resolve the full path
            full_path = Path(image_path)
            if not full_path.is_absolute():
                # Assume relative to project root
                full_path = Path("/Users/lab/Documents/vsm-v02") / image_path
            
            if not full_path.exists():
                logger.warning(f"Image not found: {full_path}")
                interpretations.append({
                    "page_number": page_num,
                    "asset_manual": manual,
                    "error": f"Image not found: {image_path}",
                })
                continue
            
            try:
                yield Status(f"Interpreting page {page_num}...")
                
                # Build a context-aware prompt
                full_prompt = (
                    f"This is page {page_num} from a technical manual called '{manual}'. "
                    f"{prompt}"
                )
                
                interpretation = await vlm_client.interpret_image(
                    image_path=str(full_path),
                    prompt=full_prompt,
                    max_tokens=512,
                )
                
                interpretations.append({
                    "page_number": page_num,
                    "asset_manual": manual,
                    "image_path": image_path,
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
        from api.services.llm import get_mlx_vlm_client
        
        diagram_type = inputs.get("diagram_type", "auto")
        extract_connections = inputs.get("extract_connections", True)
        
        yield Status("Analyzing diagram structure...")
        
        # Get visual interpretation results
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
                message="No ColQwen results to extract diagrams from",
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
        
        vlm_client = get_mlx_vlm_client()
        
        if not await vlm_client.is_available():
            yield Error(
                message="MLX VLM server is not available",
                recoverable=True,
            )
            return
        
        # Process first page with diagram content
        extractions = []
        for name, entries in colqwen_data.items():
            for entry in entries:
                for obj in entry.get("objects", [])[:1]:  # Just first page
                    image_path = obj.get("image_path")
                    if not image_path:
                        continue
                    
                    full_path = Path("/Users/lab/Documents/vsm-v02") / image_path
                    if not full_path.exists():
                        continue
                    
                    try:
                        extraction = await vlm_client.interpret_image(
                            image_path=str(full_path),
                            prompt=prompt,
                            max_tokens=1024,
                        )
                        
                        extractions.append({
                            "page_number": obj.get("page_number"),
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

