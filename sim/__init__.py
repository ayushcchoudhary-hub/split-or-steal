"""Layer 3: Configurable 3-Round Escalating Split or Steal Simulation Engine."""

from sim.backends import (
    BUILTIN_MOCK_POLICIES,
    MockPolicyBackend,
    ModelBackend,
    OpenAICompatibleBackend,
)
from sim.engine import SplitOrStealEngine, resolve_payout
from sim.metrics import AnalyticsReport, BehaviorMetrics, compute_metrics, format_scorecard
from sim.models import (
    DecisionRecord,
    ExperimentRecord,
    GameConfig,
    GameRecord,
    LLMResponse,
    PlayerConfig,
    RoundRecord,
    SPLIT,
    STEAL,
    TurnRecord,
    VALID_ACTIONS,
)
from sim.parser import (
    ParsedDecision,
    ParsedNegotiation,
    analyze_speech_intent,
    detect_cooperation_promise,
    extract_thought_and_visible,
    parse_decision_output,
    parse_negotiation_output,
)
from sim.storage import SimulationStorage

__all__ = [
    "AnalyticsReport",
    "BUILTIN_MOCK_POLICIES",
    "BehaviorMetrics",
    "DecisionRecord",
    "ExperimentRecord",
    "GameConfig",
    "GameRecord",
    "LLMResponse",
    "MockPolicyBackend",
    "ModelBackend",
    "OpenAICompatibleBackend",
    "ParsedDecision",
    "ParsedNegotiation",
    "PlayerConfig",
    "RoundRecord",
    "SPLIT",
    "STEAL",
    "SimulationStorage",
    "SplitOrStealEngine",
    "TurnRecord",
    "VALID_ACTIONS",
    "analyze_speech_intent",
    "compute_metrics",
    "detect_cooperation_promise",
    "extract_thought_and_visible",
    "format_scorecard",
    "parse_decision_output",
    "parse_negotiation_output",
    "resolve_payout",
]
