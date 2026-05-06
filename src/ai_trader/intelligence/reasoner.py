from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ai_trader.config import AppSettings, get_settings
from ai_trader.domain.signals import SignalBundle
from ai_trader.intelligence.guardrails import ReasoningGuardrails
from ai_trader.intelligence.models import NarrativeIntelligence, InsiderNewsCorrelation, InsiderNewsAlignment
from ai_trader.intelligence.trade_plan import HorizonClass, TradePlan, horizon_class_for_days
from ai_trader.llm import get_llm_client
from ai_trader.llm.contracts import LLMClient
from ai_trader.llm.errors import LLMError
from ai_trader.rag.trader_rag import TraderRAG, format_retrieved, get_trader_rag
from ai_trader.training.calibrator import LocalCalibratorModel


SYSTEM_PROMPT = (
    "You are a trading system 'final reasoner'. "
    "You fuse multiple signals and narrative context into an auditable trade plan. "
    "Pay special attention to the InsiderNewsCorrelation section: when congressional or "
    "lobbying insider trades are confirmed by global news catalysts, this is a high-signal "
    "confluence. When they contradict each other, be conservative (direction='neutral' or "
    "lower conviction). "
    "Be conservative when inputs disagree or are stale. "
    "Return only the requested JSON."
)
logger = logging.getLogger(__name__)


class FinalReasoner:
    """Phase 5-ish: Fuse signals into a structured TradePlan."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        settings: AppSettings | None = None,
        model: str | None = None,
        rag: TraderRAG | None = None,
        guardrails: ReasoningGuardrails | None = None,
        calibrator: LocalCalibratorModel | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or get_llm_client(self._settings)
        self._model = model or self._settings.llm_model
        self._rag = rag
        self._guardrails = guardrails or ReasoningGuardrails()
        self._calibrator = calibrator if calibrator is not None else _load_local_calibrator(self._settings)
        self._horizon_calibrators = (
            {} if calibrator is not None else _load_horizon_calibrators(self._settings)
        )

    def reason(
        self,
        *,
        ticker: str,
        as_of: date,
        bundle: SignalBundle,
        narrative: NarrativeIntelligence | None = None,
        position_context: str | None = None,
    ) -> TradePlan:
        parser = PydanticOutputParser(pydantic_object=TradePlan)
        narrative_json = narrative.model_dump_json(indent=2) if narrative is not None else ""
        position_text = position_context or ""
        analogies_text = ""
        if self._settings.rag_enabled:
            rag = self._rag or get_trader_rag(self._settings)
            retrieved = rag.retrieve(self._rag_query(bundle, narrative), k=3)
            analogies_text = format_retrieved(retrieved)

        correlation = self._correlate_insider_news(bundle, narrative)
        correlation_json = correlation.model_dump_json(indent=2)

        prompt = PromptTemplate(
            template=(
                "TASK: Produce a trade plan for the ticker using only the provided signals/context.\n"
                "Ticker: {ticker}\n"
                "As-of date: {as_of}\n\n"
                "SignalBundle JSON:\n{bundle_json}\n\n"
                "NarrativeIntelligence JSON (optional):\n{narrative_json}\n\n"
                "InsiderNewsCorrelation (IMPORTANT — cross-reference congressional/lobbying trades with news):\n"
                "{correlation_json}\n\n"
                "Retrieved Analogies (optional):\n{analogies_text}\n\n"
                "Current position context (optional): {position_context}\n\n"
                "Rules:\n"
                "- If insider trades and news narrative are STRONGLY_ALIGNED, this is a high-conviction confluence — you may increase conviction.\n"
                "- If insider trades and news narrative are OPPOSED or STRONGLY_OPPOSED, be conservative: use direction='neutral' or reduce conviction.\n"
                "- If there is no insider signal (has_insider_signal=false), rely on other signals as usual.\n"
                "- If evidence is weak or contradictory, use direction='neutral' with low conviction.\n"
                "- conviction is 0-1.\n"
                "- size_multiplier is 0-2 (0 means no trade).\n"
                "- holding_period_days should reflect the signal horizons.\n"
                "- exit_trigger must be specific and falsifiable.\n\n"
                "{format_instructions}"
            ),
            input_variables=[
                "ticker",
                "as_of",
                "bundle_json",
                "narrative_json",
                "correlation_json",
                "analogies_text",
                "position_context",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        text = self._invoke(
            prompt.format(
                ticker=ticker,
                as_of=as_of.isoformat(),
                bundle_json=bundle.model_dump_json(indent=2),
                narrative_json=narrative_json,
                correlation_json=correlation_json,
                analogies_text=analogies_text,
                position_context=position_text,
            )
        )
        plan = _parse(parser, text)
        guarded = self._guardrails.apply(plan=plan, bundle=bundle, narrative=narrative)

        # Apply insider-news conviction delta on top of guardrails
        if correlation.has_insider_signal and correlation.conviction_delta != 0.0:
            guarded = _apply_conviction_delta(guarded, correlation.conviction_delta)
            logger.debug(
                "Insider-news correlation adjustment: %s alignment, delta=%+.2f → conviction %.2f",
                correlation.alignment.value,
                correlation.conviction_delta,
                guarded.conviction,
            )

        dominant_horizon = _dominant_signal_horizon_days(bundle, guarded.holding_period_days)
        guarded = guarded.model_copy(
            update={"horizon_class": horizon_class_for_days(dominant_horizon)}
        )

        calibrator = self._horizon_calibrators.get(guarded.horizon_class) or self._calibrator
        if calibrator is None:
            return guarded
        return calibrator.apply(plan=guarded, bundle=bundle, narrative=narrative)

    @staticmethod
    def _correlate_insider_news(
        bundle: SignalBundle,
        narrative: NarrativeIntelligence | None,
    ) -> InsiderNewsCorrelation:
        """Cross-reference congressional/lobbying insider signals with the news narrative.

        Uses rule-based logic (no LLM call) to keep it fast and deterministic.
        The conviction_delta is then applied additively after the LLM produces its plan.
        """
        # Detect insider signals in the bundle
        insider_signal_names = {
            "lobbying_activity",
            "congressional_insider",
            "insider_trade",
            "insider_buy",
            "insider_sell",
        }
        insider_signals = [s for s in bundle.signals if s.name in insider_signal_names]

        if not insider_signals:
            return InsiderNewsCorrelation(
                has_insider_signal=False,
                insider_direction="none",
                news_sentiment="unknown",
                alignment=InsiderNewsAlignment.MIXED,
                conviction_delta=0.0,
                rationale="No congressional or lobbying signals detected in this bundle.",
            )

        # Determine net insider direction
        net = sum(s.direction.multiplier * s.strength for s in insider_signals)
        insider_direction = "buy" if net > 0 else ("sell" if net < 0 else "none")

        # Determine news sentiment from narrative
        news_sentiment = "unknown"
        surprise_dir = ""
        psych_stage = ""
        catalysts: list[str] = []

        if narrative is not None:
            surprise_dir = narrative.surprise.direction.lower()
            psych_stage = narrative.behavior.psychology_stage.value.lower()

            if surprise_dir in ("positive", "bullish", "upside"):
                news_sentiment = "positive"
            elif surprise_dir in ("negative", "bearish", "downside"):
                news_sentiment = "negative"
            elif surprise_dir in ("neutral", "in-line", "inline"):
                news_sentiment = "neutral"
            else:
                # Fall back to psychology stage
                bullish_stages = {"hope", "optimism", "belief", "thrill", "euphoria"}
                bearish_stages = {"anxiety", "denial", "panic", "capitulation", "depression"}
                if psych_stage in bullish_stages:
                    news_sentiment = "positive"
                elif psych_stage in bearish_stages:
                    news_sentiment = "negative"
                else:
                    news_sentiment = "neutral"

            what_changed = list(narrative.surprise.what_changed)
            catalysts = what_changed[:3]

        # Map (insider_direction, news_sentiment) → alignment + delta
        alignment_map: dict[tuple[str, str], tuple[InsiderNewsAlignment, float]] = {
            ("buy",  "positive"): (InsiderNewsAlignment.STRONGLY_ALIGNED, +0.15),
            ("buy",  "neutral"):  (InsiderNewsAlignment.ALIGNED,          +0.05),
            ("buy",  "negative"): (InsiderNewsAlignment.OPPOSED,          -0.15),
            ("buy",  "unknown"):  (InsiderNewsAlignment.MIXED,             0.0),
            ("sell", "negative"): (InsiderNewsAlignment.STRONGLY_ALIGNED, +0.15),
            ("sell", "neutral"):  (InsiderNewsAlignment.ALIGNED,          +0.05),
            ("sell", "positive"): (InsiderNewsAlignment.OPPOSED,          -0.15),
            ("sell", "unknown"):  (InsiderNewsAlignment.MIXED,             0.0),
            ("none", "positive"): (InsiderNewsAlignment.MIXED,             0.0),
            ("none", "negative"): (InsiderNewsAlignment.MIXED,             0.0),
            ("none", "neutral"):  (InsiderNewsAlignment.MIXED,             0.0),
            ("none", "unknown"):  (InsiderNewsAlignment.MIXED,             0.0),
        }
        alignment, delta = alignment_map.get(
            (insider_direction, news_sentiment),
            (InsiderNewsAlignment.MIXED, 0.0),
        )

        # Check for particularly strong opposition (very high surprise_score + opposite direction)
        if (
            narrative is not None
            and alignment == InsiderNewsAlignment.OPPOSED
            and narrative.surprise.surprise_score >= 0.7
        ):
            alignment = InsiderNewsAlignment.STRONGLY_OPPOSED
            delta = -0.25

        signal_names = [s.name for s in insider_signals]
        rationale = (
            f"Insider signals detected: {', '.join(signal_names)}. "
            f"Net insider direction: {insider_direction}. "
            f"News sentiment: {news_sentiment} "
            f"(surprise_dir='{surprise_dir}', psychology='{psych_stage}'). "
            f"Alignment: {alignment.value}. "
            f"Conviction delta: {delta:+.2f}."
        )

        return InsiderNewsCorrelation(
            has_insider_signal=True,
            insider_direction=insider_direction,
            news_sentiment=news_sentiment,
            alignment=alignment,
            conviction_delta=delta,
            rationale=rationale,
            catalysts=tuple(catalysts),
        )

    def _invoke(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt, system=SYSTEM_PROMPT, model=self._model, temperature=0.2)
        except Exception as exc:
            raise LLMError(f"Final reasoner LLM call failed: {exc}") from exc

    @staticmethod
    def _rag_query(bundle: SignalBundle, narrative: NarrativeIntelligence | None) -> str:
        signals = ", ".join(f"{s.name}:{s.direction.value}:{s.strength:.2f}" for s in bundle.signals)
        narrative_hint = ""
        if narrative is not None:
            narrative_hint = (
                f"Surprise={narrative.surprise.direction} "
                f"score={narrative.surprise.surprise_score:.2f} "
                f"psychology={narrative.behavior.psychology_stage.value}"
            )
        return f"{bundle.ticker} trade decision analogies. Signals: {signals}. {narrative_hint}".strip()


def _apply_conviction_delta(plan: TradePlan, delta: float) -> TradePlan:
    """Return a copy of plan with conviction clamped to [0, 1] after applying delta."""
    new_conviction = max(0.0, min(1.0, plan.conviction + delta))
    return plan.model_copy(update={"conviction": round(new_conviction, 4)})


def _parse(parser: PydanticOutputParser, text: str) -> TradePlan:
    try:
        return parser.parse(text)
    except Exception:
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            try:
                return parser.pydantic_object.model_validate_json(extracted)
            except Exception as exc:  # pragma: no cover
                raise LLMError(f"Failed to parse TradePlan JSON: {exc}") from exc
        raise LLMError("Failed to parse trade plan output as structured JSON.")


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    json.loads(candidate)
                except Exception:
                    return None
                return candidate
    return None


def _load_local_calibrator(settings: AppSettings) -> LocalCalibratorModel | None:
    if not settings.local_training_enabled:
        return None
    path = settings.local_calibrator_path
    if not path.exists():
        logger.warning("local training enabled but calibrator does not exist: %s", path)
        return None
    try:
        return LocalCalibratorModel.load(path)
    except Exception as exc:
        logger.warning("failed to load local calibrator from %s: %s", path, exc)
        return None


def _load_horizon_calibrators(settings: AppSettings) -> dict[HorizonClass, LocalCalibratorModel]:
    if not settings.local_training_enabled:
        return {}
    paths: dict[HorizonClass, Path] = {
        "short": settings.local_calibrator_short_path,
        "medium": settings.local_calibrator_medium_path,
        "long": settings.local_calibrator_long_path,
    }
    calibrators: dict[HorizonClass, LocalCalibratorModel] = {}
    for horizon, path in paths.items():
        if not path.exists():
            continue
        try:
            calibrators[horizon] = LocalCalibratorModel.load(path)
        except Exception as exc:
            logger.warning("failed to load %s horizon calibrator from %s: %s", horizon, path, exc)
    return calibrators


def _dominant_signal_horizon_days(bundle: SignalBundle, fallback: int) -> int:
    if not bundle.signals:
        return fallback
    dominant = max(
        bundle.signals,
        key=lambda signal: signal.strength * signal.confidence,
    )
    return dominant.horizon_days
