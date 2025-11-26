#!/usr/bin/env python3
"""
Fetch a GitHub repository and generate llms.txt using DSPy.

Based on the approach from docs/llm-txt-dspy.md - programmatically fetches
repo structure, README, and key files, then uses DSPy to generate documentation.

Usage:
    python fetch_repo_llms_txt.py                           # Default: DSPy repo
    python fetch_repo_llms_txt.py --repo stanfordnlp/dspy   # Specify repo
    python fetch_repo_llms_txt.py --output dspy_llms.txt    # Custom output
"""

import os
import sys
import argparse
import base64
import requests
from pathlib import Path

try:
    import dspy
except ImportError:
    print("❌ DSPy not installed. Install with: pip install dspy-ai")
    sys.exit(1)


# ==================== GitHub API Functions ====================

def get_github_file_tree(owner: str, repo: str, branch: str = "main") -> str:
    """
    Get repository file structure from GitHub API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        branch: Branch name (default: main)
    
    Returns:
        Text representation of file tree
    """
    token = os.environ.get("GITHUB_ACCESS_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        tree_data = response.json()
        file_paths = [item['path'] for item in tree_data.get('tree', []) if item['type'] == 'blob']
        return '\n'.join(sorted(file_paths))
    else:
        raise Exception(f"Failed to fetch repository tree: {response.status_code} - {response.text}")


def get_github_file_content(owner: str, repo: str, file_path: str) -> str:
    """
    Get specific file content from GitHub.
    
    Args:
        owner: Repository owner
        repo: Repository name
        file_path: Path to file in repo
    
    Returns:
        File content as string
    """
    token = os.environ.get("GITHUB_ACCESS_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        content = base64.b64decode(response.json()['content']).decode('utf-8')
        return content
    else:
        return f"Could not fetch {file_path}"


def gather_repository_info(owner: str, repo: str) -> dict:
    """
    Gather all necessary repository information.
    
    Args:
        owner: Repository owner
        repo: Repository name
    
    Returns:
        Dictionary with file_tree, readme_content, package_files
    """
    print(f"📂 Fetching file tree for {owner}/{repo}...")
    file_tree = get_github_file_tree(owner, repo)
    
    print("📄 Fetching README...")
    readme_content = get_github_file_content(owner, repo, "README.md")
    
    # Get key package files
    print("⚙️  Fetching package files...")
    package_files = []
    for file_path in ["pyproject.toml", "setup.py", "requirements.txt", "package.json"]:
        content = get_github_file_content(owner, repo, file_path)
        if "Could not fetch" not in content:
            package_files.append(f"=== {file_path} ===\n{content}")
    
    package_files_content = "\n\n".join(package_files) if package_files else "No package files found"
    
    return {
        "file_tree": file_tree,
        "readme_content": readme_content,
        "package_files": package_files_content,
    }


# ==================== DSPy Signatures ====================

class AnalyzeRepository(dspy.Signature):
    """Analyze a repository structure and identify key components."""
    repo_url: str = dspy.InputField(desc="GitHub repository URL")
    file_tree: str = dspy.InputField(desc="Repository file structure")
    readme_content: str = dspy.InputField(desc="README.md content")
    
    project_purpose: str = dspy.OutputField(desc="Main purpose and goals of the project")
    key_concepts: list[str] = dspy.OutputField(desc="List of important concepts and terminology")
    architecture_overview: str = dspy.OutputField(desc="High-level architecture description")


class AnalyzeCodeStructure(dspy.Signature):
    """Analyze code structure to identify important directories and files."""
    file_tree: str = dspy.InputField(desc="Repository file structure")
    package_files: str = dspy.InputField(desc="Key package and configuration files")
    
    important_directories: list[str] = dspy.OutputField(desc="Key directories and their purposes")
    entry_points: list[str] = dspy.OutputField(desc="Main entry points and important files")
    development_info: str = dspy.OutputField(desc="Development setup and workflow information")


class GenerateLLMsTxt(dspy.Signature):
    """Generate a comprehensive llms.txt file from analyzed repository information."""
    project_purpose: str = dspy.InputField()
    key_concepts: list[str] = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    important_directories: list[str] = dspy.InputField()
    entry_points: list[str] = dspy.InputField()
    development_info: str = dspy.InputField()
    usage_examples: str = dspy.InputField(desc="Common usage patterns and examples")
    
    llms_txt_content: str = dspy.OutputField(desc="Complete llms.txt file content following the standard format")


# ==================== DSPy Module ====================

class RepositoryAnalyzer(dspy.Module):
    """DSPy module for analyzing a repository and generating llms.txt."""
    
    def __init__(self):
        super().__init__()
        self.analyze_repo = dspy.ChainOfThought(AnalyzeRepository)
        self.analyze_structure = dspy.ChainOfThought(AnalyzeCodeStructure)
        self.generate_examples = dspy.ChainOfThought("repo_info -> usage_examples")
        self.generate_llms_txt = dspy.ChainOfThought(GenerateLLMsTxt)
    
    def forward(self, repo_url, file_tree, readme_content, package_files):
        # Analyze repository purpose and concepts
        repo_analysis = self.analyze_repo(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content,
        )
        
        # Analyze code structure
        structure_analysis = self.analyze_structure(
            file_tree=file_tree,
            package_files=package_files,
        )
        
        # Generate usage examples
        usage_examples = self.generate_examples(
            repo_info=f"Purpose: {repo_analysis.project_purpose}\nConcepts: {repo_analysis.key_concepts}"
        )
        
        # Generate final llms.txt
        llms_txt = self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples.usage_examples,
        )
        
        return dspy.Prediction(
            llms_txt_content=llms_txt.llms_txt_content,
            analysis=repo_analysis,
            structure=structure_analysis,
        )


# ==================== Main ====================

def generate_llms_txt_for_repo(owner: str, repo: str, verbose: bool = False) -> str:
    """
    Generate llms.txt for a GitHub repository.
    
    Args:
        owner: Repository owner
        repo: Repository name
        verbose: Print detailed progress
    
    Returns:
        Generated llms.txt content
    """
    repo_url = f"https://github.com/{owner}/{repo}"
    
    # Configure DSPy with local Ollama
    print("⚙️  Configuring DSPy with Ollama...")
    lm = dspy.LM(
        model="ollama/gpt-oss:120b",
        api_base="http://localhost:11434",
        api_key="dummy",
        max_tokens=4096,
    )
    dspy.configure(lm=lm)
    
    # Gather repository info
    print(f"\n🔍 Analyzing repository: {repo_url}\n")
    repo_info = gather_repository_info(owner, repo)
    
    # Truncate file tree if too long
    file_tree = repo_info["file_tree"]
    if len(file_tree) > 50000:
        lines = file_tree.split('\n')
        file_tree = '\n'.join(lines[:500]) + f"\n... ({len(lines) - 500} more files)"
        print(f"⚠️  File tree truncated to 500 entries")
    
    # Initialize analyzer
    print("🤖 Initializing analyzer...")
    analyzer = RepositoryAnalyzer()
    
    # Generate llms.txt
    print("✍️  Generating llms.txt (this may take several minutes)...\n")
    
    try:
        result = analyzer(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=repo_info["readme_content"][:30000],  # Truncate if huge
            package_files=repo_info["package_files"][:10000],
        )
        
        if verbose:
            print("\n📋 Analysis Summary:")
            print(f"   Purpose: {result.analysis.project_purpose[:100]}...")
            print(f"   Key concepts: {len(result.analysis.key_concepts)} identified")
        
        return result.llms_txt_content
    
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print("\n   Troubleshooting:")
        print("   - Is Ollama running? Check: curl http://localhost:11434/api/tags")
        print("   - Is gpt-oss:120b loaded? Run: ollama pull gpt-oss:120b")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate llms.txt for a GitHub repository using DSPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_repo_llms_txt.py                           # DSPy repo (default)
  python fetch_repo_llms_txt.py --repo stanfordnlp/dspy   # Explicit repo
  python fetch_repo_llms_txt.py --repo langchain-ai/langchain --output langchain_llms.txt

Environment:
  GITHUB_ACCESS_TOKEN - Optional, for higher API rate limits
  Ollama must be running on http://localhost:11434
        """,
    )
    
    parser.add_argument(
        "--repo", "-r",
        type=str,
        default="stanfordnlp/dspy",
        help="GitHub repo in 'owner/repo' format (default: stanfordnlp/dspy)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: {repo}_llms.txt)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Parse repo
    if "/" not in args.repo:
        print(f"❌ Invalid repo format: {args.repo}")
        print("   Use 'owner/repo' format, e.g., 'stanfordnlp/dspy'")
        sys.exit(1)
    
    owner, repo = args.repo.split("/", 1)
    
    # Generate
    content = generate_llms_txt_for_repo(owner, repo, verbose=args.verbose)
    
    if not content:
        print("❌ No content generated")
        sys.exit(1)
    
    # Save
    output_path = args.output or f"{repo}_llms.txt"
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"\n✅ Generated llms.txt successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Content size: {len(content)} characters, {len(content.splitlines())} lines")


if __name__ == "__main__":
    main()

