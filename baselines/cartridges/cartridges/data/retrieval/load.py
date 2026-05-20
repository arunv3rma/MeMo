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

if __name__ == "__main__":
    resources = get_all_resources()
    
    print(f"Collected {len(resources)} resources:")
    for resource_name, resource_content in resources.items():
        print(f"-- {resource_name}: {len(resource_content)} characters")