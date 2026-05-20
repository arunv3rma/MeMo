from typing import List, Literal, Optional
from pydantic import BaseModel

from cartridges.data.tools import Tool, ToolInput, ToolOutput
from cartridges.data.retrieval.retrievers import Retriever

import fitz
import requests
from bs4 import BeautifulSoup


database_path = "/shared/amdgpu/home/tech_ops_amd_xqh/simran/code-memory/codemem/knowledge_library/"


def extract_text_from_local_pdf(filepath: str) -> str:
    """Extract full text from a local PDF file."""
    try:
        with fitz.open(filepath) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Failed to extract PDF: {e}")
        return ""

def extract_text_from_url(url: str) -> str:
    """Extract raw visible text from an HTML page."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        print(f"Failed to extract text from {url}: {e}")
        return ""

def get_isa_manual_section(pdf_path: str, start_marker: str, end_marker: str) -> str:
    """Extracts a specific section from the ISA PDF."""
    full_text = extract_text_from_local_pdf(pdf_path)
    start = full_text.find(start_marker)
    end = full_text.find(end_marker)
    if start != -1 and end != -1:
        return full_text[start:end + len(end_marker)]
    return ""

def get_all_resources():

    resources = {}

    # Resource 1: CDNA4 whitepaper – local or remote
    resources["cdna4_whitepaper_bank_conflicts"] = extract_text_from_local_pdf(
        f"{database_path}/amd-cdna-4-architecture-whitepaper.pdf"
    )

    # Resource 2: MI300 ISA – extract specific section
    resources["mi300_vmem_section"] = get_isa_manual_section(
        f"{database_path}/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf",
        start_marker="Vector Memory (VMEM) instructions read or write",
        end_marker="and the per-wave allocation offset (also initialized in an SGPR)."
    )

    # Resource 3: HIP kernel from GitHub
    resources["fp8_hip_kernel"] = extract_text_from_url(
        "https://github.com/seb-v/amd_challenge_solutions/tree/main/fp8_gemm"
    )

    # Resource 4: Composable Kernel project
    resources["composable_kernel_info"] = extract_text_from_url(
        "https://github.com/ROCm/composable_kernel"
    )

    # Resource 5: ROCProfiler CDNA architecture info
    resources["rocprof_shader_engine"] = extract_text_from_url(
        "https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.2/conceptual/shader-engine.html"
    )

    # Resource 6 & 7: Leimao blog posts
    resources["bank_conflicts_leimao"] = extract_text_from_url(
        "https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/"
    )

    resources["swizzling_leimao"] = extract_text_from_url(
        "https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/"
    )

    # Resource 8: AMD HIP performance guide
    resources["hip_perf_guide"] = extract_text_from_url(
        "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html"
    )

    return resources


class SourceConfig(BaseModel):
    path: str
    type: Literal["local_pdf", "url", "isa_manual_section"] = "url"
    start_marker: Optional[str] = None
    end_marker: Optional[str] = None

def load_source(source: SourceConfig) -> str:
    if source.type == "local_pdf":
        return extract_text_from_local_pdf(source.path)
    elif source.type == "url":
        return extract_text_from_url(source.path)
    elif source.type == "isa_manual_section":
        if source.start_marker is None or source.end_marker is None:
            raise ValueError("start_marker and end_marker must be provided for isa_manual_section")
        return get_isa_manual_section(source.path, source.start_marker, source.end_marker)
    else:
        raise ValueError(f"Invalid source type: {source.type}")

class RetrievalTool(Tool):
    class Config(Tool.Config):
        retriever: Retriever.Config
        sources: List[SourceConfig]
    
    class ToolInput(ToolInput):
        query: str
        top_k: int = 1

    def __init__(self, config: Config):
        super().__init__(config)

        print("Loading sources...")
        sources = [load_source(source) for source in config.sources]
        print(f"Loaded {len(sources)} sources")

        self.retriever = config.retriever.instantiate(
            sources=sources
        )

    async def run_tool(self, input: ToolInput) -> ToolOutput:
        return ToolOutput(
            input=input,
            success=True,
            error=None,
            response=await self.retriever.retrieve(query=input.query, top_k=input.top_k)
        )
    
    @property
    def description(self) -> str:
        return "Retrieve relevant information from AMD, HIP, and ThunderKittens documentation."
    
    @property
    def name(self) -> str:
        return "retrieve"

AMD_TK_SOURCES = [
    SourceConfig(
        path=f"{database_path}/amd-cdna-4-architecture-whitepaper.pdf",
        type="local_pdf",
        start_marker="Vector Memory (VMEM) instructions read or write",
        end_marker="and the per-wave allocation offset (also initialized in an SGPR)."
    ),
    SourceConfig(
        path=f"{database_path}/amd-cdna-4-architecture-whitepaper.pdf",
        type="local_pdf"
    ),
    SourceConfig(
        path=f"{database_path}/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf",
        type="local_pdf"
    ),
    SourceConfig(
        path="https://github.com/seb-v/amd_challenge_solutions/tree/main/fp8_gemm",
        type="url"
    ),
    SourceConfig(
        path="https://github.com/ROCm/composable_kernel",
        type="url"
    ),
    SourceConfig(
        path="https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.2/conceptual/shader-engine.html",
        type="url"
    ),
    SourceConfig(
        path="https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/",
        type="url"
    ),
    SourceConfig(
        path="https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/",
        type="url"
    ),
]