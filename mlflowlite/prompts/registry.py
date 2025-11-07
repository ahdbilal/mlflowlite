"""
Prompt registry using MLflow's native Prompt Registry.

This provides a wrapper around MLflow's prompt management features:
- Register prompts with mlflow.register_prompt()
- Load prompts with mlflow.load_prompt()
- Version control with commit messages
- View prompts in MLflow UI's Prompts tab

See: https://mlflow.org/docs/2.21.3/prompts
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import mlflow


@dataclass
class PromptVersion:
    """A versioned prompt (for backwards compatibility)."""
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
    """
    Registry for managing prompt versions using MLflow's native Prompt Registry.
    
    This class wraps MLflow's prompt management to provide:
    - Automatic registration in MLflow's Prompts tab
    - Version control with commit messages
    - Easy loading and formatting of prompts
    
    See: https://mlflow.org/docs/2.21.3/prompts
    """
    
    def __init__(
        self,
        agent_name: str,
        registry_path: Optional[str] = None,
    ):
        """
        Initialize prompt registry.
        
        Args:
            agent_name: Name of the agent (used as prompt name prefix)
            registry_path: Legacy parameter (kept for backwards compatibility)
        """
        self.agent_name = agent_name
        
        # Sanitize prompt name for Databricks/MLflow (only alphanumeric and underscores)
        # Remove all invalid characters and ensure it starts with a letter
        sanitized_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in agent_name)
        # Remove consecutive underscores and strip trailing underscores
        sanitized_name = '_'.join(filter(None, sanitized_name.split('_')))
        # Ensure it starts with a letter (prepend 'agent' if it starts with number/underscore)
        if not sanitized_name or not sanitized_name[0].isalpha():
            sanitized_name = f"agent_{sanitized_name}"
        # Convert to lowercase (Databricks MLflow requires lowercase names)
        sanitized_name = sanitized_name.lower()
        # Limit length to 64 characters (Databricks limit)
        sanitized_name = sanitized_name[:50]  # Leave room for '_prompt'
        
        # Detect if running on Databricks and get Unity Catalog schema
        self.is_databricks = self._is_databricks()
        self.uc_schema = self._get_uc_schema()
        
        # Format prompt name based on environment
        if self.is_databricks and self.uc_schema:
            # Databricks Unity Catalog format: catalog.schema.prompt_name
            self.prompt_name = f"{self.uc_schema}.{sanitized_name}_prompt"
        else:
            # Local format: simple name
            self.prompt_name = f"{sanitized_name}_prompt"
        
        self.current_version = 0
        
        # Set up local storage only if NOT on Databricks UC
        if not (self.is_databricks and self.uc_schema):
            # Local mode: use file-based fallback
            if registry_path:
                self.registry_path = Path(registry_path)
            else:
                self.registry_path = Path.home() / ".mlflowlite" / "prompts" / agent_name
            
            self.registry_path.mkdir(parents=True, exist_ok=True)
            self.registry_file = self.registry_path / "registry.json"
            self.versions: List[PromptVersion] = []
        else:
            # Databricks UC mode: no local storage
            self.registry_path = None
            self.registry_file = None
            self.versions = []
        
        # Try to load existing prompts from MLflow
        try:
            client = mlflow.MlflowClient()
            mlflow_versions = client.search_prompt_versions(f"name='{self.prompt_name}'")
            if mlflow_versions:
                self.current_version = max([v.version for v in mlflow_versions])
        except Exception:
            # MLflow prompts not available
            if not (self.is_databricks and self.uc_schema):
                # Local mode: load from local file as fallback
                self._load_registry()
        
        # Initialize with default prompt if empty (only in local mode)
        if self.current_version == 0 and not self.versions and not (self.is_databricks and self.uc_schema):
            self._initialize_default_prompt()
    
    def _is_databricks(self) -> bool:
        """Check if running on Databricks."""
        try:
            import IPython
            ipython = IPython.get_ipython()
            if ipython and 'DATABRICKS' in str(type(ipython)):
                return True
        except:
            pass
        
        # Check environment variables
        return any([
            'DATABRICKS_RUNTIME_VERSION' in os.environ,
            'DATABRICKS_HOST' in os.environ,
            os.path.exists('/databricks/spark/conf')
        ])
    
    def _get_uc_schema(self) -> Optional[str]:
        """
        Get Unity Catalog schema for prompts.
        
        Priority:
        1. MLflow experiment tag mlflow.promptRegistryLocation
        2. Environment variable MLFLOW_PROMPT_REGISTRY_UC_SCHEMA
        3. Default: main.default (if on Databricks)
        """
        if not self.is_databricks:
            return None
        
        # Try to get from MLflow experiment tag
        try:
            active_run = mlflow.active_run()
            if active_run:
                experiment = mlflow.get_experiment(active_run.info.experiment_id)
                if experiment and experiment.tags:
                    uc_schema = experiment.tags.get('mlflow.promptRegistryLocation')
                    if uc_schema:
                        return uc_schema
        except:
            pass
        
        # Try environment variable
        uc_schema = os.environ.get('MLFLOW_PROMPT_REGISTRY_UC_SCHEMA')
        if uc_schema:
            return uc_schema
        
        # Default to main.default for Databricks
        return 'main.default'
    
    def _load_registry(self):
        """Load registry from disk (legacy fallback)."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.versions = [PromptVersion.from_dict(v) for v in data.get("versions", [])]
                    if self.versions:
                        self.current_version = self.versions[-1].version
            except Exception as e:
                print(f"Warning: Failed to load registry: {e}")
                self.versions = []
    
    def _save_registry(self):
        """Save registry to disk (legacy fallback - skipped in Databricks UC mode)."""
        # Skip local storage when using Databricks Unity Catalog
        if self.is_databricks and self.uc_schema:
            return
        
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
        default_system = """You are a helpful AI agent with access to tools.

Your goal is to assist users by:
1. Understanding their request
2. Breaking it down into steps
3. Using available tools when needed
4. Providing clear, accurate responses

Think step-by-step and explain your reasoning when using tools."""
        
        self.add_version(
            system_prompt=default_system,
            user_template="{query}",
            examples=[],
            metadata={"source": "default", "auto_generated": True}
        )
    
    def add_version(
        self,
        system_prompt: str,
        user_template: str = "{query}",
        examples: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """
        Add a new prompt version using MLflow's register_prompt().
        
        This registers the prompt in MLflow and makes it visible in the Prompts tab!
        
        Args:
            system_prompt: System prompt text
            user_template: Template for user messages (can use {query} placeholder)
            examples: Optional few-shot examples
            metadata: Additional metadata
        
        Returns:
            The new prompt version
        """
        version_num = self.current_version + 1
        
        # Build the combined template for MLflow
        template_parts = [f"System: {system_prompt}"]
        
        if examples:
            template_parts.append("\nExamples:")
            for ex in examples or []:
                template_parts.append(f"User: {ex.get('input', '')}")
                template_parts.append(f"Assistant: {ex.get('output', '')}")
        
        # Convert {query} to {{query}} for MLflow's double-brace format
        mlflow_template = user_template.replace('{', '{{').replace('}', '}}')
        template_parts.append(f"\nUser: {mlflow_template}")
        
        full_template = "\n".join(template_parts)
        
        # Try to register with MLflow
        try:
            commit_msg = metadata.get('change', 'Updated prompt') if metadata else 'New prompt version'
            
            # Register prompt with MLflow
            # Use appropriate API based on environment
            import mlflow as mlflow_lib
            
            if self.is_databricks:
                # Databricks: Use mlflow.genai API with Unity Catalog
                try:
                    from mlflow import genai as mlflow_genai
                    # Sanitize metadata for Unity Catalog
                    safe_tags = {k.replace('-', '_').replace('.', '_'): str(v)
                               for k, v in (metadata or {}).items()}
                    safe_tags.update({
                        "agent": self.agent_name[:50],
                        "type": "agent_prompt"
                    })
                    
                    registered_prompt = mlflow_genai.register_prompt(
                        name=self.prompt_name,  # Full UC path: catalog.schema.prompt_name
                        template=full_template,
                        commit_message=commit_msg,
                        tags=safe_tags
                    )
                except Exception as e:
                    # If Unity Catalog registration fails, let it fall through to local storage
                    raise e
            else:
                # Local: Use legacy mlflow.register_prompt API
                try:
                    safe_metadata = {k.replace('-', '_').replace('.', '_'): v 
                                   for k, v in (metadata or {}).items()}
                    registered_prompt = mlflow_lib.register_prompt(
                        name=self.prompt_name,
                        template=full_template,
                        commit_message=commit_msg,
                        version_metadata=safe_metadata,
                        tags={
                            "agent": self.agent_name[:50],
                            "type": "agent_prompt"
                        }
                    )
                except TypeError:
                    # Fall back to basic API without optional parameters
                    registered_prompt = mlflow_lib.register_prompt(
                        name=self.prompt_name,
                        template=full_template,
                    )
            
            self.current_version = registered_prompt.version
            print(f"✅ Registered prompt '{self.prompt_name}' version {self.current_version} in MLflow")
            print(f"   View in MLflow UI: Prompts tab → {self.prompt_name}")
            
        except Exception as e:
            print(f"Warning: Failed to register prompt with MLflow: {e}")
            print("Using local storage as fallback...")
        
        # Create version object (for backwards compatibility and local storage)
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
        """
        Get a specific prompt version (tries MLflow first, then local).
        
        Args:
            version: Version number to retrieve
            
        Returns:
            PromptVersion object or None if not found
        """
        # Try MLflow first
        try:
            prompt = mlflow.load_prompt(f"prompts:/{self.prompt_name}/{version}")
            
            # Parse the template back into components
            template_text = prompt.template
            system_part = template_text.split("User:")[0].replace("System:", "").strip()
            user_part = template_text.split("User:")[-1].strip() if "User:" in template_text else "{query}"
            
            return PromptVersion(
                version=version,
                system_prompt=system_part,
                user_template=user_part.replace('{{', '{').replace('}}', '}'),  # Convert back to single braces
                examples=[],
                metadata={},
                created_at=datetime.now().isoformat(),
            )
        except Exception:
            # Databricks UC mode: no local fallback
            if self.is_databricks and self.uc_schema:
                return None
            
            # Local mode: fall back to local storage
            for v in self.versions:
                if v.version == version:
                    return v
            return None
    
    def get_latest(self) -> PromptVersion:
        """Get the latest prompt version."""
        if self.current_version > 0:
            latest = self.get_version(self.current_version)
            if latest:
                return latest
        
        # Databricks UC mode: no local fallback
        if self.is_databricks and self.uc_schema:
            raise ValueError(
                f"No prompt versions found in MLflow for '{self.prompt_name}'. "
                f"Please register a prompt first using add_version()."
            )
        
        # Local mode: fallback to local storage
        if not self.versions:
            self._initialize_default_prompt()
        return self.versions[-1]
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions (tries MLflow first, then local).
        
        Returns:
            List of version info dictionaries
        """
        # Try MLflow first
        try:
            client = mlflow.MlflowClient()
            mlflow_versions = client.search_prompt_versions(f"name='{self.prompt_name}'")
            
            return [
                {
                    'version': v.version,
                    'created_at': datetime.now().isoformat(),
                    'metadata': v.version_metadata or {},
                    'commit_message': getattr(v, 'commit_message', ''),
                }
                for v in mlflow_versions
            ]
        except Exception:
            # Databricks UC mode: no local fallback
            if self.is_databricks and self.uc_schema:
                return []  # No versions found in MLflow
            
            # Local mode: fall back to local storage
            return [
                {
                    "version": v.version,
                    "created_at": v.created_at,
                    "system_prompt_preview": v.system_prompt[:100] + "..." if len(v.system_prompt) > 100 else v.system_prompt,
                    "num_examples": len(v.examples),
                    "metadata": v.metadata,
                }
                for v in self.versions
            ]
    
    def update_performance(self, version: int, metrics: Dict[str, float]):
        """Update performance metrics for a version."""
        prompt_version = self.get_version(version)
        if prompt_version and self.versions:
            for v in self.versions:
                if v.version == version:
                    v.performance_metrics = metrics
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


def load_prompt(name: str, version: Optional[int] = None) -> str:
    """
    Load a prompt from MLflow Prompt Registry.
    
    Clean, simple API for retrieving prompts:
    
    Args:
        name: Prompt name (e.g., "sentiment_classifier")
        version: Optional version number. If None, loads latest.
    
    Returns:
        Prompt text (system_prompt)
    
    Examples:
        >>> # Load latest version
        >>> prompt = load_prompt("sentiment_classifier")
        >>> 
        >>> # Load specific version
        >>> prompt = load_prompt("sentiment_classifier", version=1)
        >>> 
        >>> # Use inline with Agent
        >>> result = Agent(
        ...     model="claude-sonnet-4-20250514",
        ...     prompt=load_prompt("sentiment_classifier", version=1)
        ... )(text="Great!")
    """
    registry = PromptRegistry(agent_name=name)
    
    if version is None:
        # Load latest
        prompt_version = registry.get_latest()
    else:
        # Load specific version
        prompt_version = registry.get_version(version)
        if not prompt_version:
            raise ValueError(f"Version {version} not found for prompt '{name}'")
    
    return prompt_version.system_prompt
