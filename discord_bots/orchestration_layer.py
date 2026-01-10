"""
Tiered LLM Orchestration Layer for RALPH Agent Ensemble

This module provides a cost-efficient orchestration layer that uses
cheap/fast models (GPT-4o-mini or Claude Haiku) for simple operations
and only escalates to Claude Code for complex tasks.

Token Savings Strategy:
- Task Classification: Determine if task needs Claude Code (~90% cheaper)
- Routing: Decide which agent should handle (no Claude needed)
- Simple Responses: Handle acknowledgments, status checks locally
- Context Summarization: Compress long contexts before passing to Claude

Supported Models:
- OpenAI: gpt-5-mini (cheaper and better than gpt-4o-mini)
- Anthropic: claude-3-haiku (~$0.25/1M input, $1.25/1M output)
- Local: Pattern matching for very simple tasks (free)
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("orchestration")


def load_openai_key_from_polymarket_config() -> Optional[str]:
    """
    Load OpenAI API key from Polymarket AI Bot's config.yaml.

    Checks POLYMARKET_PROJECT_DIR env var for the project location.
    """
    polymarket_dir = os.getenv("POLYMARKET_PROJECT_DIR")
    if not polymarket_dir:
        return None

    config_path = Path(polymarket_dir) / "config.yaml"
    if not config_path.exists():
        return None

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Check common config key paths
        if isinstance(config, dict):
            # Try different possible locations for the key
            if "openai" in config and "api_key" in config["openai"]:
                return config["openai"]["api_key"]
            if "openai_api_key" in config:
                return config["openai_api_key"]
            if "OPENAI_API_KEY" in config:
                return config["OPENAI_API_KEY"]
            if "api_keys" in config and "openai" in config["api_keys"]:
                return config["api_keys"]["openai"]

    except ImportError:
        logger.debug("PyYAML not installed, can't read config.yaml")
    except Exception as e:
        logger.debug(f"Failed to read Polymarket config: {e}")

    return None


class TaskComplexity(Enum):
    """Classification of task complexity."""
    TRIVIAL = "trivial"      # Can be handled locally with pattern matching
    SIMPLE = "simple"        # Cheap LLM can handle (questions, routing, summaries)
    MODERATE = "moderate"    # Claude Sonnet - straightforward code tasks
    COMPLEX = "complex"      # Claude Opus 4.5 - architecture, multi-file, deep reasoning


class ClaudeModel(Enum):
    """Claude models for different task complexities."""
    SONNET = "claude-sonnet-4-20250514"      # Fast, good for routine tasks
    OPUS = "claude-opus-4-5-20250101"        # Best reasoning, for complex tasks


# Map complexity to recommended Claude model
COMPLEXITY_TO_MODEL = {
    TaskComplexity.MODERATE: ClaudeModel.SONNET,
    TaskComplexity.COMPLEX: ClaudeModel.OPUS,
}


class OrchestratorProvider(Enum):
    """Supported orchestration LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # Pattern matching only


@dataclass
class ClassificationResult:
    """Result of task classification."""
    complexity: TaskComplexity
    confidence: float  # 0-1
    reasoning: str
    suggested_agent: Optional[str] = None
    can_handle_locally: bool = False
    local_response: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result of orchestration operation."""
    handled: bool  # True if orchestrator handled it, False if needs Claude Code
    response: Optional[str] = None
    route_to_agent: Optional[str] = None
    summarized_context: Optional[str] = None
    tokens_used: int = 0
    cost_estimate: float = 0.0
    recommended_model: Optional[str] = None  # Claude model to use if not handled


class TaskClassifier:
    """
    Classifies incoming tasks to determine handling strategy.

    Uses pattern matching for trivial tasks, cheap LLM for ambiguous ones.
    """

    # Patterns for trivial tasks (can be handled locally)
    TRIVIAL_PATTERNS = {
        # Status checks
        r"^(status|ping|health|alive)\?*$": ("trivial_status", "System is operational."),
        r"^what('s| is) (your |the )?(status|state)\?*$": ("trivial_status", None),

        # Simple acknowledgments
        r"^(ok|okay|yes|no|thanks|thank you|got it|understood)\!*$": ("ack", None),

        # Help requests (return help text)
        r"^help\!*$": ("help", None),
        r"^what can you do\?*$": ("help", None),

        # Time/date (no LLM needed)
        r"^what time is it\?*$": ("time", None),
        r"^what('s| is) (the )?date\?*$": ("date", None),
    }

    # Patterns indicating complex tasks (need Claude Code)
    COMPLEX_INDICATORS = [
        r"(write|create|implement|build|develop|code|program)",
        r"(edit|modify|change|update|fix|refactor|optimize) .*(file|code|function|class)",
        r"(analyze|review|audit) .*(code|implementation|architecture)",
        r"(run|execute) .*(test|backtest|simulation)",
        r"(deploy|rollback|migrate)",
        r"multi-?file",
        r"(database|schema|migration)",
        r"(git|commit|push|pull|merge|rebase)",
    ]

    # Patterns indicating simple tasks (cheap LLM can handle)
    SIMPLE_INDICATORS = [
        r"^(what|who|when|where|why|how) ",  # Questions
        r"(explain|describe|summarize|list)",
        r"(should|could|would|can) (i|we|you)",
        r"(recommend|suggest|advise)",
        r"^is (it|this|that) ",
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self._trivial_compiled = {
            re.compile(pattern, re.IGNORECASE): (task_type, response)
            for pattern, (task_type, response) in self.TRIVIAL_PATTERNS.items()
        }
        self._complex_compiled = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.COMPLEX_INDICATORS
        ]
        self._simple_compiled = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SIMPLE_INDICATORS
        ]

    def classify_local(self, task: str) -> ClassificationResult:
        """
        Classify task using only local pattern matching.
        Fast and free, but limited accuracy.
        """
        task_clean = task.strip().lower()

        # Check trivial patterns first
        for pattern, (task_type, response) in self._trivial_compiled.items():
            if pattern.match(task_clean):
                return ClassificationResult(
                    complexity=TaskComplexity.TRIVIAL,
                    confidence=0.95,
                    reasoning=f"Matched trivial pattern: {task_type}",
                    can_handle_locally=True,
                    local_response=response
                )

        # Count complex indicators
        complex_matches = sum(
            1 for pattern in self._complex_compiled
            if pattern.search(task)
        )

        # Count simple indicators
        simple_matches = sum(
            1 for pattern in self._simple_compiled
            if pattern.search(task)
        )

        # Heuristic classification
        if complex_matches >= 2:
            return ClassificationResult(
                complexity=TaskComplexity.COMPLEX,
                confidence=0.7 + (0.1 * min(complex_matches, 3)),
                reasoning=f"Found {complex_matches} complex indicators"
            )
        elif complex_matches == 1 and simple_matches == 0:
            return ClassificationResult(
                complexity=TaskComplexity.MODERATE,
                confidence=0.6,
                reasoning="Found 1 complex indicator, needs LLM classification"
            )
        elif simple_matches >= 1:
            return ClassificationResult(
                complexity=TaskComplexity.SIMPLE,
                confidence=0.6 + (0.1 * min(simple_matches, 3)),
                reasoning=f"Found {simple_matches} simple indicators"
            )
        else:
            # Ambiguous - needs LLM to classify
            return ClassificationResult(
                complexity=TaskComplexity.MODERATE,
                confidence=0.4,
                reasoning="No clear indicators, needs LLM classification"
            )


class ContextSummarizer:
    """
    Summarizes long contexts to reduce token usage.

    Uses cheap LLM to compress context while preserving key information.
    """

    # Maximum context length before summarization (in characters)
    MAX_CONTEXT_LENGTH = 4000

    # Target summary length
    TARGET_SUMMARY_LENGTH = 1000

    SUMMARIZE_PROMPT = """Summarize the following context for an AI agent.
Preserve:
- Key decisions made
- Important values/metrics
- Action items
- Critical warnings or errors

Context to summarize:
{context}

Provide a concise summary (max 500 words):"""

    def needs_summarization(self, context: str) -> bool:
        """Check if context needs summarization."""
        return len(context) > self.MAX_CONTEXT_LENGTH

    def extract_key_info(self, context: str) -> Dict[str, Any]:
        """
        Extract key information from context using pattern matching.
        This is free and fast.
        """
        info = {
            "metrics": [],
            "errors": [],
            "decisions": [],
            "action_items": []
        }

        # Extract metrics (numbers with labels)
        metric_pattern = r"(\w+[\w\s]*?):\s*([\d.]+%?|\$[\d,.]+)"
        for match in re.finditer(metric_pattern, context):
            info["metrics"].append(f"{match.group(1)}: {match.group(2)}")

        # Extract errors/warnings
        error_pattern = r"(error|warning|failed|exception)[:\s]+([^\n]+)"
        for match in re.finditer(error_pattern, context, re.IGNORECASE):
            info["errors"].append(match.group(0))

        # Extract action items
        action_pattern = r"(TODO|FIXME|ACTION|NEXT)[:\s]+([^\n]+)"
        for match in re.finditer(action_pattern, context, re.IGNORECASE):
            info["action_items"].append(match.group(2))

        return info


class AgentRouter:
    """
    Routes tasks to the appropriate agent.

    Uses pattern matching and cheap LLM for routing decisions.
    """

    # Keywords that indicate specific agents
    AGENT_KEYWORDS = {
        "tuning": [
            "parameter", "hyperparameter", "optimize", "tune", "learning rate",
            "threshold", "coefficient", "weight", "regularization", "epoch"
        ],
        "backtest": [
            "backtest", "simulate", "historical", "validate", "test",
            "performance", "sharpe", "drawdown", "returns", "pnl"
        ],
        "risk": [
            "risk", "safety", "limit", "exposure", "veto", "audit",
            "compliance", "drawdown limit", "position size", "stop loss"
        ],
        "strategy": [
            "strategy", "signal", "entry", "exit", "logic", "rule",
            "indicator", "momentum", "mean reversion", "trend"
        ],
        "data": [
            "data", "feature", "preprocess", "clean", "normalize",
            "pipeline", "etl", "database", "api", "websocket"
        ]
    }

    def route_local(self, task: str, from_agent: str = None) -> Optional[str]:
        """
        Route task to agent using keyword matching.
        Returns None if routing is ambiguous.
        """
        task_lower = task.lower()

        # Count keyword matches per agent
        scores = {}
        for agent, keywords in self.AGENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[agent] = score

        if not scores:
            return None

        # If clear winner (2x score of second), route there
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) == 1:
            return sorted_scores[0][0]

        top_agent, top_score = sorted_scores[0]
        second_agent, second_score = sorted_scores[1]

        if top_score >= second_score * 2:
            return top_agent

        # Ambiguous - needs LLM
        return None


class OrchestrationLayer:
    """
    Main orchestration layer that coordinates all components.

    Supports multiple LLM providers for the cheap/fast model.
    """

    def __init__(
        self,
        provider: OrchestratorProvider = None,
        model: str = None
    ):
        # Try to get OpenAI key from multiple sources
        self._openai_key = os.getenv("OPENAI_API_KEY")
        if not self._openai_key:
            # Try loading from Polymarket AI Bot's config.yaml
            self._openai_key = load_openai_key_from_polymarket_config()
            if self._openai_key:
                logger.info("Loaded OpenAI API key from Polymarket config.yaml")

        # Auto-detect provider based on available API keys
        if provider is None:
            if self._openai_key:
                provider = OrchestratorProvider.OPENAI
                model = model or "gpt-5-mini"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = OrchestratorProvider.ANTHROPIC
                model = model or "claude-3-haiku-20240307"
            else:
                provider = OrchestratorProvider.LOCAL
                model = None

        self.provider = provider
        self.model = model

        # Initialize components
        self.classifier = TaskClassifier()
        self.summarizer = ContextSummarizer()
        self.router = AgentRouter()

        # Initialize LLM client
        self._client = None
        self._init_client()

        # Stats tracking
        self.stats = {
            "tasks_handled_locally": 0,
            "tasks_handled_cheap_llm": 0,
            "tasks_escalated_to_claude": 0,
            "tokens_saved_estimate": 0,
            "cost_saved_estimate": 0.0
        }

        logger.info(f"OrchestrationLayer initialized with provider: {provider.value}, model: {model}")

    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == OrchestratorProvider.OPENAI:
            try:
                from openai import AsyncOpenAI
                # Use the key we found (from env or Polymarket config)
                self._client = AsyncOpenAI(api_key=self._openai_key)
            except ImportError:
                logger.warning("OpenAI package not installed. Install with: pip install openai")
                self.provider = OrchestratorProvider.LOCAL

        elif self.provider == OrchestratorProvider.ANTHROPIC:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                logger.warning("Anthropic package not installed. Install with: pip install anthropic")
                self.provider = OrchestratorProvider.LOCAL

    async def _call_cheap_llm(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 500
    ) -> Tuple[str, int]:
        """
        Call the cheap LLM and return (response, tokens_used).
        """
        if self.provider == OrchestratorProvider.LOCAL:
            return "", 0

        try:
            if self.provider == OrchestratorProvider.OPENAI:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=0.3
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens
                return content, tokens

            elif self.provider == OrchestratorProvider.ANTHROPIC:
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
                return content, tokens

        except Exception as e:
            logger.error(f"Error calling cheap LLM: {e}")
            return "", 0

        return "", 0

    async def classify_task(
        self,
        task: str,
        context: str = ""
    ) -> ClassificationResult:
        """
        Classify a task to determine handling strategy.

        Uses local classification first, then cheap LLM if ambiguous.
        """
        # Try local classification first (free)
        result = self.classifier.classify_local(task)

        # If high confidence or trivial, return immediately
        if result.confidence >= 0.8 or result.complexity == TaskComplexity.TRIVIAL:
            return result

        # If local-only mode, return what we have
        if self.provider == OrchestratorProvider.LOCAL:
            return result

        # Use cheap LLM for ambiguous cases
        classify_prompt = f"""Classify this task for an AI agent system.

Task: {task}

Classify as one of:
- TRIVIAL: Simple status check, acknowledgment, or predefined response
- SIMPLE: Question answering, explanation, or routing decision
- MODERATE: Requires analysis but no code changes
- COMPLEX: Requires writing/editing code, multi-step implementation

Also suggest which agent should handle it:
- tuning: Parameter optimization
- backtest: Testing and validation
- risk: Safety and compliance
- strategy: Trading logic
- data: Data pipeline

Respond in JSON format:
{{"complexity": "SIMPLE|MODERATE|COMPLEX", "agent": "agent_name", "reasoning": "brief explanation"}}"""

        response, tokens = await self._call_cheap_llm(
            classify_prompt,
            system="You are a task classifier. Respond only with valid JSON.",
            max_tokens=150
        )

        if response:
            try:
                data = json.loads(response)
                return ClassificationResult(
                    complexity=TaskComplexity[data.get("complexity", "MODERATE")],
                    confidence=0.85,
                    reasoning=data.get("reasoning", "LLM classification"),
                    suggested_agent=data.get("agent")
                )
            except (json.JSONDecodeError, KeyError):
                pass

        return result

    async def route_task(
        self,
        task: str,
        from_agent: str = None
    ) -> Optional[str]:
        """
        Determine which agent should handle a task.

        Returns agent name or None if ambiguous.
        """
        # Try local routing first
        agent = self.router.route_local(task, from_agent)
        if agent:
            return agent

        # Use cheap LLM if local is ambiguous
        if self.provider != OrchestratorProvider.LOCAL:
            route_prompt = f"""Which agent should handle this task?

Task: {task}

Agents:
- tuning: Parameter optimization, hyperparameter search
- backtest: Testing, validation, performance analysis
- risk: Safety audits, compliance, risk assessment
- strategy: Trading logic, signals, entry/exit rules
- data: Data pipeline, preprocessing, feature engineering

Respond with just the agent name (tuning/backtest/risk/strategy/data):"""

            response, _ = await self._call_cheap_llm(
                route_prompt,
                system="You are a task router. Respond with only the agent name.",
                max_tokens=20
            )

            if response:
                agent = response.strip().lower()
                if agent in ["tuning", "backtest", "risk", "strategy", "data"]:
                    return agent

        return None

    async def summarize_context(
        self,
        context: str,
        focus: str = ""
    ) -> str:
        """
        Summarize long context to reduce tokens.

        Returns original context if short enough.
        """
        if not self.summarizer.needs_summarization(context):
            return context

        # Extract key info locally first
        key_info = self.summarizer.extract_key_info(context)

        # Build summary prompt
        focus_hint = f"\nFocus on information relevant to: {focus}" if focus else ""

        summarize_prompt = f"""Summarize this context concisely for an AI agent.{focus_hint}

Key extracted info:
- Metrics: {', '.join(key_info['metrics'][:5]) or 'None'}
- Errors: {', '.join(key_info['errors'][:3]) or 'None'}
- Action items: {', '.join(key_info['action_items'][:3]) or 'None'}

Full context:
{context[:6000]}

Provide a 200-word summary preserving critical information:"""

        if self.provider != OrchestratorProvider.LOCAL:
            response, _ = await self._call_cheap_llm(
                summarize_prompt,
                system="You are a context summarizer. Be concise but preserve key information.",
                max_tokens=400
            )

            if response:
                return response

        # Fallback: truncate with key info
        summary_parts = ["**Key Information:**"]
        if key_info["metrics"]:
            summary_parts.append(f"Metrics: {', '.join(key_info['metrics'][:5])}")
        if key_info["errors"]:
            summary_parts.append(f"Errors: {', '.join(key_info['errors'][:3])}")
        if key_info["action_items"]:
            summary_parts.append(f"Actions: {', '.join(key_info['action_items'][:3])}")
        summary_parts.append(f"\n**Context (truncated):**\n{context[:2000]}...")

        return "\n".join(summary_parts)

    async def handle_simple_task(
        self,
        task: str,
        agent_context: str = ""
    ) -> OrchestrationResult:
        """
        Attempt to handle a simple task without Claude Code.

        Returns result with handled=True if successful.
        """
        classification = await self.classify_task(task)

        # Handle trivial tasks locally
        if classification.complexity == TaskComplexity.TRIVIAL:
            if classification.local_response:
                self.stats["tasks_handled_locally"] += 1
                return OrchestrationResult(
                    handled=True,
                    response=classification.local_response,
                    tokens_used=0
                )

        # Handle simple tasks with cheap LLM
        if classification.complexity == TaskComplexity.SIMPLE:
            if self.provider != OrchestratorProvider.LOCAL:
                response, tokens = await self._call_cheap_llm(
                    f"Answer this question concisely:\n\n{task}\n\nContext: {agent_context[:1000]}",
                    system="You are a helpful AI assistant. Be concise and accurate.",
                    max_tokens=500
                )

                if response:
                    self.stats["tasks_handled_cheap_llm"] += 1
                    # Estimate savings (Claude Code would use ~3-5K tokens minimum)
                    self.stats["tokens_saved_estimate"] += 4000 - tokens

                    return OrchestrationResult(
                        handled=True,
                        response=response,
                        tokens_used=tokens,
                        cost_estimate=tokens * 0.00000015  # GPT-4o-mini pricing
                    )

        # Can't handle locally - needs Claude Code
        self.stats["tasks_escalated_to_claude"] += 1

        # But we can still optimize by summarizing context
        summarized = await self.summarize_context(agent_context, focus=task)

        # Select Claude model based on task complexity and type
        recommended_model = self._select_claude_model(task, classification.complexity)

        return OrchestrationResult(
            handled=False,
            route_to_agent=classification.suggested_agent,
            summarized_context=summarized,
            recommended_model=recommended_model
        )

    def _select_claude_model(self, task: str, complexity: TaskComplexity) -> str:
        """
        Select the appropriate Claude model based on task characteristics.

        Opus 4.5 for:
        - Planning, architecture, strategy tasks
        - Complex multi-file changes
        - Deep analysis and reasoning

        Sonnet for:
        - Routine code changes
        - Simple fixes and updates
        - Running scripts/tests
        """
        task_lower = task.lower()

        # Always use Opus for planning/strategy/architecture
        planning_indicators = [
            "plan", "design", "architect", "strategy", "approach",
            "how should", "what's the best way", "recommend",
            "analyze", "investigate", "diagnose", "debug complex",
            "refactor", "redesign", "proposal", "evaluate options"
        ]

        if any(indicator in task_lower for indicator in planning_indicators):
            return ClaudeModel.OPUS.value

        # Use complexity mapping for other tasks
        model = COMPLEXITY_TO_MODEL.get(complexity, ClaudeModel.SONNET)
        return model.value

    async def process_incoming(
        self,
        task: str,
        from_agent: str = None,
        context: str = ""
    ) -> OrchestrationResult:
        """
        Main entry point for processing incoming tasks.

        1. Classifies the task
        2. Attempts to handle locally or with cheap LLM
        3. Returns handling result with optimized context
        """
        # Try to handle without Claude Code
        result = await self.handle_simple_task(task, context)

        if result.handled:
            return result

        # Route to appropriate agent if not specified
        if not result.route_to_agent:
            result.route_to_agent = await self.route_task(task, from_agent)

        return result

    def get_stats_report(self) -> str:
        """Get a report of orchestration statistics."""
        total = (
            self.stats["tasks_handled_locally"] +
            self.stats["tasks_handled_cheap_llm"] +
            self.stats["tasks_escalated_to_claude"]
        )

        if total == 0:
            return "No tasks processed yet."

        local_pct = self.stats["tasks_handled_locally"] / total * 100
        cheap_pct = self.stats["tasks_handled_cheap_llm"] / total * 100
        claude_pct = self.stats["tasks_escalated_to_claude"] / total * 100

        # Estimate cost savings (Claude Code ~$0.015 per 1K tokens)
        saved_cost = self.stats["tokens_saved_estimate"] * 0.000015

        return f"""**Orchestration Stats:**
- Total tasks: {total}
- Handled locally: {self.stats['tasks_handled_locally']} ({local_pct:.1f}%)
- Handled by cheap LLM: {self.stats['tasks_handled_cheap_llm']} ({cheap_pct:.1f}%)
- Escalated to Claude: {self.stats['tasks_escalated_to_claude']} ({claude_pct:.1f}%)
- Estimated tokens saved: {self.stats['tokens_saved_estimate']:,}
- Estimated cost saved: ${saved_cost:.4f}"""


# =============================================================================
# CONVERSATIONAL MISSION MANAGER
# =============================================================================

class ConversationalMissionManager:
    """
    Uses GPT-4o-mini to drive multi-turn conversations with Claude Code.

    This solves the problem that Claude Code with --print only gives one response.
    GPT-4o-mini acts as the "manager" that:
    1. Analyzes the mission
    2. Creates prompts for Claude Code
    3. Evaluates Claude's response
    4. Decides on follow-up prompts
    5. Continues until task is complete
    """

    def __init__(self, orchestration_layer: "OrchestrationLayer"):
        self.orch = orchestration_layer
        self.conversation_history: List[Dict[str, str]] = []
        self.max_turns = 10  # Prevent infinite loops

    async def drive_mission(
        self,
        mission_objective: str,
        project_dir: str,
        claude_executor,  # ClaudeExecutor instance
        post_update: Callable[[str], Any] = None  # Callback to post Discord updates
    ) -> Dict[str, Any]:
        """
        Drive a mission to completion using GPT as manager, Claude as worker.

        Since Claude Code --print mode doesn't reliably use tools, we:
        1. Explore the codebase ourselves (Python)
        2. Pass the file contents to Claude as context
        3. Have Claude analyze and output tasks

        Args:
            mission_objective: The high-level mission goal
            project_dir: Working directory for Claude Code
            claude_executor: ClaudeExecutor instance for running Claude Code
            post_update: Async callback to post updates to Discord

        Returns:
            Dict with tasks, status, and summary
        """
        import glob
        from pathlib import Path

        self.conversation_history = []
        tasks_found = []

        if post_update:
            await post_update("**Exploring codebase** to find relevant files...")

        # Step 1: Explore the codebase ourselves
        project_path = Path(project_dir)
        relevant_files = []

        # Search patterns for ML/trading code
        patterns = [
            "**/regime*.py", "**/*model*.py", "**/*ml*.py",
            "**/paper*.py", "**/trading*.py", "**/inference*.py",
            "**/train*.py", "**/predict*.py", "**/backtest*.py",
            "**/strategy*.py", "**/risk*.py", "**/data*.py",
            "**/config*.yaml", "**/config*.json"
        ]

        # Directories to exclude
        exclude_dirs = ["__pycache__", "venv", ".venv", "node_modules", ".git"]

        for pattern in patterns:
            matches = list(project_path.glob(pattern))
            for match in matches[:5]:  # Limit per pattern
                match_str = str(match)
                if match.is_file() and not any(ex in match_str for ex in exclude_dirs):
                    relevant_files.append(match)

        # Deduplicate
        relevant_files = list(set(relevant_files))[:15]  # Max 15 files

        if post_update:
            file_list = "\n".join([f"- {f.name}" for f in relevant_files])
            await post_update(f"**Found {len(relevant_files)} relevant files:**\n{file_list}")

        # Step 2: Read the files
        file_contents = {}
        for file_path in relevant_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                # Truncate long files
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                file_contents[str(file_path.relative_to(project_path))] = content
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")

        # Step 3: Build context with file contents
        context_parts = ["# CODEBASE ANALYSIS\n"]
        for file_name, content in file_contents.items():
            context_parts.append(f"\n## File: {file_name}\n```python\n{content}\n```\n")

        codebase_context = "".join(context_parts)

        # Step 4: Create prompt for Claude with the context
        current_prompt = f"""YOU MUST OUTPUT TASKS IN THIS EXACT FORMAT. Do not have a conversation. Do not ask questions. Just output tasks.

REQUIRED OUTPUT FORMAT (output at least 5 tasks):
[TASK: data] description referencing specific files
[TASK: tuning] description referencing specific files
[TASK: backtest] description referencing specific files
[TASK: risk] description referencing specific files
[TASK: strategy] description referencing specific files

MISSION: {mission_objective}

CODEBASE FILES:
{codebase_context}

NOW OUTPUT YOUR TASKS using the [TASK: agent] format above. Reference actual files from the codebase. Do not explain, do not ask questions, just output the tasks."""

        turn = 0
        while turn < self.max_turns:
            turn += 1

            if post_update:
                await post_update(f"**Manager Turn {turn}**: Sending prompt to Claude Code...")

            # Execute with Claude Code
            result = await claude_executor.execute(
                agent_name="Strategy Agent",
                agent_role="Mission planner and code analyst",
                task_prompt=current_prompt,
                context=None
            )

            claude_response = result.output if result.output else "No output"

            if post_update:
                # Post Claude's full response (will be chunked if needed)
                await post_update(f"**Claude Response ({result.duration_seconds:.1f}s)**:\n{claude_response}")

            # Parse any tasks from Claude's response
            task_pattern = r'\[TASK:\s*(\w+)\]\s*(.+?)(?=\[TASK:|$)'
            matches = re.findall(task_pattern, claude_response, re.IGNORECASE | re.DOTALL)

            for agent_type, task_desc in matches:
                tasks_found.append({
                    "agent": agent_type.lower().strip(),
                    "task": task_desc.strip()[:500]
                })

            # If we found tasks, mission planning is done
            if tasks_found:
                if post_update:
                    await post_update(f"**Mission Planning Complete**: Found {len(tasks_found)} tasks!")
                break

            # Use GPT to evaluate and create follow-up prompt
            evaluation = await self._evaluate_response(
                mission_objective,
                claude_response,
                turn
            )

            if evaluation["complete"]:
                break

            if evaluation["follow_up_prompt"]:
                current_prompt = evaluation["follow_up_prompt"]

                if post_update:
                    await post_update(f"**Manager**: {evaluation['reasoning']}")
            else:
                # No follow-up, but also no tasks - ask more directly
                current_prompt = f"""The previous response didn't include any [TASK:] items.

Please explore the codebase and output your implementation plan using this EXACT format:

[TASK: data] description of data-related task
[TASK: tuning] description of tuning-related task
[TASK: backtest] description of testing task
[TASK: risk] description of risk audit task
[TASK: strategy] description of strategy task

The objective is: {mission_objective}

Use Glob and Read to explore files, then output the tasks."""

        return {
            "tasks": tasks_found,
            "turns_used": turn,
            "complete": len(tasks_found) > 0,
            "summary": f"Found {len(tasks_found)} tasks in {turn} turns"
        }

    async def _evaluate_response(
        self,
        objective: str,
        claude_response: str,
        turn: int
    ) -> Dict[str, Any]:
        """
        Use GPT-4o-mini to evaluate Claude's response and decide next steps.
        """
        if self.orch.provider == OrchestratorProvider.LOCAL:
            # Fallback: simple heuristic
            has_tasks = "[TASK:" in claude_response.upper()
            return {
                "complete": has_tasks,
                "follow_up_prompt": None if has_tasks else "Please output tasks using [TASK: agent] format",
                "reasoning": "Local evaluation"
            }

        eval_prompt = f"""You are managing an AI coding assistant. Evaluate its response and decide next steps.

MISSION OBJECTIVE: {objective}

CLAUDE'S RESPONSE:
{claude_response[:2000]}

TURN: {turn} of 10

Questions to answer:
1. Did Claude explore actual files in the codebase? (Look for file paths, function names)
2. Did Claude output tasks in [TASK: agent] format?
3. Are the tasks specific enough (reference actual files/functions)?

If Claude didn't complete the mission planning, provide a follow-up prompt.

Respond in JSON:
{{
    "explored_files": true/false,
    "has_tasks": true/false,
    "tasks_are_specific": true/false,
    "complete": true/false,
    "follow_up_prompt": "prompt to send if not complete, or null",
    "reasoning": "brief explanation"
}}"""

        response, _ = await self.orch._call_cheap_llm(
            eval_prompt,
            system="You are a task manager evaluating AI responses. Respond only with valid JSON.",
            max_tokens=500
        )

        try:
            data = json.loads(response)
            return {
                "complete": data.get("complete", False),
                "follow_up_prompt": data.get("follow_up_prompt"),
                "reasoning": data.get("reasoning", "GPT evaluation")
            }
        except (json.JSONDecodeError, TypeError):
            # Fallback
            return {
                "complete": "[TASK:" in claude_response.upper(),
                "follow_up_prompt": None,
                "reasoning": "JSON parse failed, using heuristic"
            }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_orchestration_layer: Optional[OrchestrationLayer] = None


def get_orchestration_layer() -> OrchestrationLayer:
    """Get or create the orchestration layer instance."""
    global _orchestration_layer
    if _orchestration_layer is None:
        _orchestration_layer = OrchestrationLayer()
    return _orchestration_layer


def set_orchestration_layer(layer: OrchestrationLayer):
    """Set the orchestration layer instance."""
    global _orchestration_layer
    _orchestration_layer = layer
