"""Prompt registry for versioning and managing agent prompts."""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class PromptVersion:
    """A versioned prompt."""
    version: int
    system_prompt: str
    user_template: str
    examples: List[Dict[str, str]]
    metadata: Dict[str, Any]
    created_at: str
    performance_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        return cls(**data)


class PromptRegistry:
    """Registry for managing prompt versions."""
    
    def __init__(
        self,
        agent_name: str,
        registry_path: Optional[str] = None,
    ):
        """
        Initialize prompt registry.
        
        Args:
            agent_name: Name of the agent
            registry_path: Path to store registry files (defaults to ~/.mlflowlite/prompts)
        """
        self.agent_name = agent_name
        
        if registry_path:
            self.registry_path = Path(registry_path)
        else:
            self.registry_path = Path.home() / ".mlflowlite" / "prompts" / agent_name
        
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        
        # Load or initialize registry
        self.versions: List[PromptVersion] = []
        self._load_registry()
        
        # Initialize with default prompt if empty
        if not self.versions:
            self._initialize_default_prompt()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.versions = [PromptVersion.from_dict(v) for v in data.get("versions", [])]
            except Exception as e:
                print(f"Warning: Failed to load registry: {e}")
                self.versions = []
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {
                "agent_name": self.agent_name,
                "versions": [v.to_dict() for v in self.versions],
            }
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save registry: {e}")
    
    def _initialize_default_prompt(self):
        """Initialize with a default prompt."""
        default_prompt = PromptVersion(
            version=1,
            system_prompt="""You are a helpful AI agent with access to tools.

Your goal is to assist users by:
1. Understanding their request
2. Breaking it down into steps
3. Using available tools when needed
4. Providing clear, accurate responses

Think step-by-step and explain your reasoning when using tools.""",
            user_template="{query}",
            examples=[],
            metadata={"source": "default", "auto_generated": True},
            created_at=datetime.now().isoformat(),
        )
        
        self.versions.append(default_prompt)
        self._save_registry()
    
    def add_version(
        self,
        system_prompt: str,
        user_template: str = "{query}",
        examples: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """
        Add a new prompt version.
        
        Args:
            system_prompt: System prompt text
            user_template: Template for user messages
            examples: Few-shot examples
            metadata: Additional metadata
        
        Returns:
            The new prompt version
        """
        version_num = len(self.versions) + 1
        
        new_version = PromptVersion(
            version=version_num,
            system_prompt=system_prompt,
            user_template=user_template,
            examples=examples or [],
            metadata=metadata or {},
            created_at=datetime.now().isoformat(),
        )
        
        self.versions.append(new_version)
        self._save_registry()
        
        return new_version
    
    def get_version(self, version: int) -> Optional[PromptVersion]:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version:
                return v
        return None
    
    def get_latest(self) -> PromptVersion:
        """Get the latest prompt version."""
        if not self.versions:
            self._initialize_default_prompt()
        return self.versions[-1]
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions with summary info."""
        summaries = []
        for v in self.versions:
            summary = {
                "version": v.version,
                "created_at": v.created_at,
                "system_prompt_preview": v.system_prompt[:100] + "..." if len(v.system_prompt) > 100 else v.system_prompt,
                "num_examples": len(v.examples),
                "metadata": v.metadata,
            }
            if v.performance_metrics:
                summary["performance"] = v.performance_metrics
            summaries.append(summary)
        return summaries
    
    def update_performance(self, version: int, metrics: Dict[str, float]):
        """Update performance metrics for a version."""
        prompt_version = self.get_version(version)
        if prompt_version:
            prompt_version.performance_metrics = metrics
            self._save_registry()
    
    def compare(self, v1: int, v2: int) -> Dict[str, Any]:
        """Compare two prompt versions."""
        version1 = self.get_version(v1)
        version2 = self.get_version(v2)
        
        if not version1 or not version2:
            raise ValueError(f"Invalid version numbers: {v1}, {v2}")
        
        return {
            "version_1": v1,
            "version_2": v2,
            "system_prompt_diff": {
                "v1_length": len(version1.system_prompt),
                "v2_length": len(version2.system_prompt),
                "changed": version1.system_prompt != version2.system_prompt,
            },
            "examples_diff": {
                "v1_count": len(version1.examples),
                "v2_count": len(version2.examples),
            },
            "performance_diff": {
                "v1": version1.performance_metrics or {},
                "v2": version2.performance_metrics or {},
            },
        }
    
    def rollback(self, version: int):
        """
        Rollback to a previous version by making it the latest.
        
        Args:
            version: Version number to rollback to
        """
        old_version = self.get_version(version)
        if not old_version:
            raise ValueError(f"Version {version} not found")
        
        # Create a new version with the old prompt
        self.add_version(
            system_prompt=old_version.system_prompt,
            user_template=old_version.user_template,
            examples=old_version.examples,
            metadata={
                **old_version.metadata,
                "rollback_from": version,
                "rollback_at": datetime.now().isoformat(),
            }
        )
    
    def export_version(self, version: int, output_path: str):
        """Export a prompt version to a file."""
        prompt_version = self.get_version(version)
        if not prompt_version:
            raise ValueError(f"Version {version} not found")
        
        with open(output_path, 'w') as f:
            json.dump(prompt_version.to_dict(), f, indent=2)
    
    def import_version(self, input_path: str) -> PromptVersion:
        """Import a prompt version from a file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        return self.add_version(
            system_prompt=data["system_prompt"],
            user_template=data["user_template"],
            examples=data.get("examples", []),
            metadata={
                **data.get("metadata", {}),
                "imported_at": datetime.now().isoformat(),
            }
        )

