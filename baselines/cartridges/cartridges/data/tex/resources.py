import asyncio
from typing import List, Optional

import os
import re
import argparse
import tempfile
import tarfile
import aiohttp

from cartridges.data.chunkers import Chunker, TokenChunker
from cartridges.data.resources import Resource, SEED_TYPES, sample_seed_prompts

class LaTeXResource(Resource):
    
    class Config(Resource.Config):
        # The root of the tex project
        root_dir: Optional[str] = None
        arxiv_id: Optional[str] = None
        
        # The relative path to the main tex file
        tex_file: str="main.tex"
        
        chunker: Chunker.Config
        
        seed_prompts: List[SEED_TYPES]

    def __init__(self, config: Config):
        self.config = config
        assert (self.config.root_dir is None) != (self.config.arxiv_id is None)
        self.chunker = None
        
    
    async def setup(self):
        root_dir = self.config.root_dir
        
        if self.config.arxiv_id is not None:
            if ("arxiv.org" in self.config.arxiv_id):
                arxiv_id = self.config.arxiv_id.split("/")[-1]
            else:
                arxiv_id = self.config.arxiv_id

            # Download and extract arXiv source
            root_dir = await self._download_arxiv_source(arxiv_id)

        tex = await process_latex_project(root_dir)
        self.tex = tex
        self.chunker = self.config.chunker.instantiate(text=tex)
        return tex
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        if self.chunker is None:
            raise ValueError("Chunker not initialized. Call setup() first.")
        
        chunk = self.chunker.sample_chunk()
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return chunk, seed_prompts


    async def _download_arxiv_source(self, arxiv_id: str) -> str:
        """Download arXiv source and extract to temporary directory."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"arxiv_{arxiv_id}_")
        
        # Download source
        source_url = f"https://arxiv.org/src/{arxiv_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(source_url) as response:
                response.raise_for_status()
                content = await response.read()
        
        # Save to temporary file asynchronously
        tar_path = os.path.join(temp_dir, f"{arxiv_id}.tar.gz")
        
        def write_file():
            with open(tar_path, 'wb') as f:
                f.write(content)
        
        await asyncio.to_thread(write_file)
        
        # Extract tar file (this part remains sync as tarfile doesn't have async support)
        extract_dir = os.path.join(temp_dir, "extracted")
        
        def make_dirs():
            os.makedirs(extract_dir, exist_ok=True)
        
        await asyncio.to_thread(make_dirs)
        
        def extract_tar():
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        
        await asyncio.to_thread(extract_tar)
        
        # Remove tar file to save space
        await asyncio.to_thread(os.remove, tar_path)
        
        return extract_dir
        



def remove_latex_comments(text):
    """
    Remove LaTeX comments from text.
    A LaTeX comment starts with % and continues to the end of the line,
    unless the % is escaped or part of a command/environment.
    """
    # This is a simple implementation that may not handle all edge cases
    # (like % inside verbatim environments)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Find % that are not preceded by \
        comment_pos = -1
        pos = line.find('%')
        while pos != -1:
            # Check if % is escaped
            if pos > 0 and line[pos-1] == '\\':
                pos = line.find('%', pos + 1)
            else:
                comment_pos = pos
                break
        
        if comment_pos != -1:
            # Keep only the part before the comment
            line = line[:comment_pos].rstrip()
        else:
            line = line.rstrip()
        if len(line.strip()) > 0:
            cleaned_lines.append(line)

    
    return '\n'.join(cleaned_lines)


def find_tex_files(directory):
    """Find all .tex files in the directory and its subdirectories."""
    tex_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
    
    return tex_files


async def process_latex_project(directory: str) -> str:
    """
    Process all .tex files in the directory, remove comments,
    and concatenate them into a single text file.
    """
    tex_files = find_tex_files(directory)
    
    if not tex_files:
        print(f"No .tex files found in {directory}")
        return ""
    
    combined_content = []
    
    for tex_file in sorted(tex_files):
        print(f"Processing: {tex_file}")
        
        def read_file():
            with open(tex_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        content = await asyncio.to_thread(read_file)
        
        # Remove comments and whitespace
        clean_content = remove_latex_comments(content)
        
        # Add file separator
        file_relative_path = os.path.relpath(tex_file, directory)
        combined_content.append(f"\n\n% Content from: {file_relative_path}\n")
        combined_content.append(clean_content)
    
    # Write to output file
    return '\n'.join(combined_content)


if __name__ == "__main__":
    
    resource = LaTeXResource.Config(
        arxiv_id="2506.06266",
        chunker=TokenChunker.Config(
            tokenizer="Qwen/Qwen3-4b",
            min_tokens_per_chunk=512,
            max_tokens_per_chunk=1024,
        ),
        seed_prompts=["generic"]
    ).instantiate()

    async def main():
        await resource.setup()
        for i in range(10):
            ctx, seed_prompts = await resource.sample_prompt(1)
            print(ctx)
            print(seed_prompts)
            print("-"*100)

    asyncio.run(main())
    
