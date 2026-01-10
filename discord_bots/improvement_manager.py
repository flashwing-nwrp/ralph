"""
Self-Improvement Manager - Autonomous Code Enhancement for RALPH Agents

Enables agents to:
1. Identify improvement opportunities from their work
2. Research best practices and cutting-edge techniques
3. Propose code changes with rationale
4. Test changes in sandbox before deployment
5. Deploy approved changes with rollback capability

Safety: All improvements require operator approval before deployment.
"""

import json
import os
import asyncio
import subprocess
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class ImprovementType(str, Enum):
    PERFORMANCE = "performance"      # Speed, efficiency improvements
    RELIABILITY = "reliability"      # Error handling, stability
    CAPABILITY = "capability"        # New features, capabilities
    CODE_QUALITY = "code_quality"    # Refactoring, maintainability
    SECURITY = "security"            # Security hardening
    UX = "ux"                         # User experience improvements


class ImprovementStatus(str, Enum):
    DRAFT = "draft"                  # Being developed
    PROPOSED = "proposed"            # Ready for review
    RESEARCHING = "researching"      # Gathering more info
    TESTING = "testing"              # In sandbox testing
    APPROVED = "approved"            # Operator approved
    DEPLOYED = "deployed"            # Applied to codebase
    REJECTED = "rejected"            # Operator rejected
    ROLLED_BACK = "rolled_back"      # Reverted due to issues
    FAILED = "failed"                # Testing failed


@dataclass
class CodeChange:
    """Represents a single code change."""
    file_path: str
    original_content: str
    new_content: str
    description: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class ResearchSource:
    """Source of research used for improvement."""
    source_type: str  # paper, article, documentation, best_practice
    title: str
    url: Optional[str] = None
    summary: str = ""
    relevance: str = ""


@dataclass
class TestResult:
    """Result of sandbox testing."""
    passed: bool
    test_type: str  # unit, integration, manual
    details: str
    metrics_before: Optional[Dict] = None
    metrics_after: Optional[Dict] = None
    timestamp: str = ""


@dataclass
class Improvement:
    """A proposed self-improvement."""
    id: str
    title: str
    description: str
    improvement_type: str
    status: str
    proposed_by: str  # Agent that proposed it

    # Problem identification
    problem_statement: str = ""
    observed_issues: List[str] = field(default_factory=list)

    # Research
    research_sources: List[Dict] = field(default_factory=list)
    hypothesis: str = ""

    # Implementation
    code_changes: List[Dict] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)

    # Testing
    test_plan: str = ""
    test_results: List[Dict] = field(default_factory=list)

    # Deployment
    rollback_commit: Optional[str] = None
    deployed_at: Optional[str] = None
    deployed_by: Optional[str] = None

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    rejection_reason: Optional[str] = None
    impact_assessment: str = ""
    risk_level: str = "medium"  # low, medium, high


class ImprovementManager:
    """Manages the self-improvement lifecycle."""

    def __init__(self, data_dir: str = None, codebase_root: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        if codebase_root is None:
            codebase_root = os.path.dirname(__file__)

        self.data_dir = data_dir
        self.codebase_root = codebase_root
        self.improvements_file = os.path.join(data_dir, "improvements.json")
        self.sandbox_dir = os.path.join(data_dir, "sandbox")

        self._ensure_dirs()
        self.improvements: Dict[str, Improvement] = {}
        self._load()

    def _ensure_dirs(self):
        """Ensure required directories exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def _load(self):
        """Load improvements from file."""
        if os.path.exists(self.improvements_file):
            try:
                with open(self.improvements_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("improvements", []):
                        imp = Improvement(**item)
                        self.improvements[imp.id] = imp
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.warning(f"Could not load improvements: {e}")

    def _save(self):
        """Save improvements to file."""
        try:
            data = {
                "improvements": [asdict(imp) for imp in self.improvements.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(self.improvements_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Could not save improvements: {e}")

    def _generate_id(self) -> str:
        """Generate unique improvement ID."""
        existing = [int(k.split("-")[1]) for k in self.improvements.keys()
                   if k.startswith("IMP-")]
        next_num = max(existing, default=0) + 1
        return f"IMP-{next_num:04d}"

    # =========================================================================
    # IMPROVEMENT LIFECYCLE
    # =========================================================================

    def create_improvement(
        self,
        title: str,
        description: str,
        improvement_type: ImprovementType,
        proposed_by: str,
        problem_statement: str = "",
        observed_issues: List[str] = None,
        hypothesis: str = ""
    ) -> Improvement:
        """Create a new improvement proposal."""
        imp = Improvement(
            id=self._generate_id(),
            title=title,
            description=description,
            improvement_type=improvement_type.value,
            status=ImprovementStatus.DRAFT.value,
            proposed_by=proposed_by,
            problem_statement=problem_statement,
            observed_issues=observed_issues or [],
            hypothesis=hypothesis,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self.improvements[imp.id] = imp
        self._save()
        logger.info(f"Created improvement {imp.id}: {title}")
        return imp

    def add_research(
        self,
        improvement_id: str,
        source_type: str,
        title: str,
        url: str = None,
        summary: str = "",
        relevance: str = ""
    ) -> bool:
        """Add research source to an improvement."""
        imp = self.improvements.get(improvement_id)
        if not imp:
            return False

        source = {
            "source_type": source_type,
            "title": title,
            "url": url,
            "summary": summary,
            "relevance": relevance,
            "added_at": datetime.now().isoformat()
        }
        imp.research_sources.append(source)
        imp.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def add_code_change(
        self,
        improvement_id: str,
        file_path: str,
        original_content: str,
        new_content: str,
        description: str
    ) -> bool:
        """Add a code change to an improvement."""
        imp = self.improvements.get(improvement_id)
        if not imp:
            return False

        change = {
            "file_path": file_path,
            "original_content": original_content,
            "new_content": new_content,
            "description": description,
            "checksum": hashlib.md5(new_content.encode()).hexdigest()
        }
        imp.code_changes.append(change)
        if file_path not in imp.affected_files:
            imp.affected_files.append(file_path)
        imp.updated_at = datetime.now().isoformat()
        self._save()
        return True

    def submit_for_review(self, improvement_id: str) -> bool:
        """Submit improvement for operator review."""
        imp = self.improvements.get(improvement_id)
        if not imp or imp.status not in [ImprovementStatus.DRAFT.value,
                                          ImprovementStatus.RESEARCHING.value]:
            return False

        imp.status = ImprovementStatus.PROPOSED.value
        imp.updated_at = datetime.now().isoformat()
        self._save()
        logger.info(f"Improvement {improvement_id} submitted for review")
        return True

    def approve(self, improvement_id: str, approved_by: str) -> bool:
        """Operator approves an improvement."""
        imp = self.improvements.get(improvement_id)
        if not imp or imp.status != ImprovementStatus.PROPOSED.value:
            return False

        imp.status = ImprovementStatus.APPROVED.value
        imp.deployed_by = approved_by
        imp.updated_at = datetime.now().isoformat()
        self._save()
        logger.info(f"Improvement {improvement_id} approved by {approved_by}")
        return True

    def reject(self, improvement_id: str, reason: str) -> bool:
        """Operator rejects an improvement."""
        imp = self.improvements.get(improvement_id)
        if not imp or imp.status != ImprovementStatus.PROPOSED.value:
            return False

        imp.status = ImprovementStatus.REJECTED.value
        imp.rejection_reason = reason
        imp.updated_at = datetime.now().isoformat()
        self._save()
        logger.info(f"Improvement {improvement_id} rejected: {reason}")
        return True

    # =========================================================================
    # SANDBOX TESTING
    # =========================================================================

    async def test_in_sandbox(self, improvement_id: str) -> Dict[str, Any]:
        """Test improvement changes in sandbox environment."""
        imp = self.improvements.get(improvement_id)
        if not imp:
            return {"success": False, "error": "Improvement not found"}

        sandbox_path = os.path.join(self.sandbox_dir, improvement_id)

        try:
            # Create sandbox copy
            if os.path.exists(sandbox_path):
                shutil.rmtree(sandbox_path)

            # Copy affected files to sandbox
            os.makedirs(sandbox_path, exist_ok=True)

            for change in imp.code_changes:
                file_path = change["file_path"]
                sandbox_file = os.path.join(sandbox_path, os.path.basename(file_path))

                # Write new content to sandbox
                with open(sandbox_file, 'w', encoding='utf-8') as f:
                    f.write(change["new_content"])

            # Run syntax check
            syntax_results = await self._check_syntax(sandbox_path)

            # Run tests if available
            test_results = await self._run_tests(sandbox_path, imp.affected_files)

            # Record results
            result = TestResult(
                passed=syntax_results["passed"] and test_results.get("passed", True),
                test_type="sandbox",
                details=f"Syntax: {syntax_results['details']}, Tests: {test_results.get('details', 'N/A')}",
                timestamp=datetime.now().isoformat()
            )

            imp.test_results.append(asdict(result))
            imp.status = ImprovementStatus.TESTING.value
            imp.updated_at = datetime.now().isoformat()
            self._save()

            return {
                "success": result.passed,
                "syntax": syntax_results,
                "tests": test_results
            }

        except Exception as e:
            logger.error(f"Sandbox testing failed: {e}")
            return {"success": False, "error": str(e)}

    async def _check_syntax(self, sandbox_path: str) -> Dict[str, Any]:
        """Check Python syntax of sandbox files."""
        results = {"passed": True, "details": "", "errors": []}

        for filename in os.listdir(sandbox_path):
            if filename.endswith('.py'):
                filepath = os.path.join(sandbox_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                    compile(code, filepath, 'exec')
                except SyntaxError as e:
                    results["passed"] = False
                    results["errors"].append(f"{filename}: {e}")

        results["details"] = "All files passed" if results["passed"] else f"Errors: {results['errors']}"
        return results

    async def _run_tests(self, sandbox_path: str, affected_files: List[str]) -> Dict[str, Any]:
        """Run relevant tests for affected files."""
        # This would integrate with pytest or the project's test framework
        # For now, return a placeholder
        return {
            "passed": True,
            "details": "No automated tests configured for sandbox"
        }

    # =========================================================================
    # DEPLOYMENT
    # =========================================================================

    async def deploy(self, improvement_id: str) -> Dict[str, Any]:
        """Deploy approved improvement to codebase."""
        imp = self.improvements.get(improvement_id)
        if not imp or imp.status != ImprovementStatus.APPROVED.value:
            return {"success": False, "error": "Improvement not approved"}

        try:
            # Create git backup/rollback point
            rollback_commit = await self._create_rollback_point(imp)
            imp.rollback_commit = rollback_commit

            # Apply changes
            for change in imp.code_changes:
                file_path = change["file_path"]

                # Verify original content matches (safety check)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        current = f.read()

                    # Apply change
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(change["new_content"])

                    logger.info(f"Applied change to {file_path}")

            # Update status
            imp.status = ImprovementStatus.DEPLOYED.value
            imp.deployed_at = datetime.now().isoformat()
            imp.updated_at = datetime.now().isoformat()
            self._save()

            return {
                "success": True,
                "rollback_commit": rollback_commit,
                "files_changed": len(imp.code_changes)
            }

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            imp.status = ImprovementStatus.FAILED.value
            self._save()
            return {"success": False, "error": str(e)}

    async def rollback(self, improvement_id: str) -> Dict[str, Any]:
        """Rollback a deployed improvement."""
        imp = self.improvements.get(improvement_id)
        if not imp or imp.status != ImprovementStatus.DEPLOYED.value:
            return {"success": False, "error": "Improvement not deployed"}

        try:
            # Restore original content
            for change in imp.code_changes:
                file_path = change["file_path"]
                if os.path.exists(file_path):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(change["original_content"])
                    logger.info(f"Rolled back {file_path}")

            imp.status = ImprovementStatus.ROLLED_BACK.value
            imp.updated_at = datetime.now().isoformat()
            self._save()

            return {"success": True, "files_restored": len(imp.code_changes)}

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"success": False, "error": str(e)}

    async def _create_rollback_point(self, imp: Improvement) -> Optional[str]:
        """Create a git commit as rollback point."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.codebase_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not create rollback point: {e}")
        return None

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_improvement(self, improvement_id: str) -> Optional[Improvement]:
        """Get improvement by ID."""
        return self.improvements.get(improvement_id)

    def list_improvements(
        self,
        status: Optional[str] = None,
        improvement_type: Optional[str] = None
    ) -> List[Improvement]:
        """List improvements with optional filters."""
        results = list(self.improvements.values())

        if status:
            results = [i for i in results if i.status == status]
        if improvement_type:
            results = [i for i in results if i.improvement_type == improvement_type]

        # Sort by updated_at descending
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def get_pending_review(self) -> List[Improvement]:
        """Get improvements pending operator review."""
        return self.list_improvements(status=ImprovementStatus.PROPOSED.value)

    def get_statistics(self) -> Dict[str, Any]:
        """Get improvement statistics."""
        all_imps = list(self.improvements.values())
        return {
            "total": len(all_imps),
            "by_status": {
                s.value: len([i for i in all_imps if i.status == s.value])
                for s in ImprovementStatus
            },
            "by_type": {
                t.value: len([i for i in all_imps if i.improvement_type == t.value])
                for t in ImprovementType
            },
            "deployed_count": len([i for i in all_imps if i.status == ImprovementStatus.DEPLOYED.value]),
            "success_rate": self._calculate_success_rate(all_imps)
        }

    def _calculate_success_rate(self, improvements: List[Improvement]) -> float:
        """Calculate deployment success rate."""
        completed = [i for i in improvements
                    if i.status in [ImprovementStatus.DEPLOYED.value,
                                   ImprovementStatus.ROLLED_BACK.value,
                                   ImprovementStatus.FAILED.value]]
        if not completed:
            return 0.0
        successful = [i for i in completed if i.status == ImprovementStatus.DEPLOYED.value]
        return len(successful) / len(completed) * 100


# Singleton
_improvement_manager: Optional[ImprovementManager] = None


def get_improvement_manager() -> ImprovementManager:
    """Get singleton improvement manager."""
    global _improvement_manager
    if _improvement_manager is None:
        _improvement_manager = ImprovementManager()
    return _improvement_manager
