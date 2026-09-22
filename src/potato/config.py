"""Application settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from potato import __version__


def _split_csv(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v and str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Client-facing — comma-separated in env (NoDecode skips JSON parsing)
    proxy_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Upstream NIM
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    nim_rpm_limit: int = 40
    nim_rpm_safety_factor: float = 0.9
    nim_cooldown_seconds: float = 60.0

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    # INVARIANT: upstream_timeout >= per_attempt_budget_seconds
    # INVARIANT: request_deadline_seconds >= upstream_timeout
    # Violating these guarantees a 504 cascade
    upstream_timeout: float = 300.0
    upstream_connect_timeout_seconds: float = 5.0
    upstream_pool_timeout_seconds: float = 10.0
    upstream_write_timeout_seconds: float = 30.0
    enable_ttft_hedging: bool = True
    ttft_hedge_factor: float = 1.8
    # P2-1: true parallel TTFT hedging (race primary vs secondary, first
    # content-bearing delta wins). Default off — canary before enabling.
    enable_parallel_hedge: bool = False
    parallel_hedge_delay_seconds: float = 2.5
    parallel_hedge_max_per_request: int = 1
    allow_graceful_fallback_on_explicit: bool = False
    default_model: str | None = None
    # Streaming: short TTFT = fail-fast to next model if not responding;
    # long idle once first token arrives (Cursor/agent safe).
    stream_ttft_timeout_seconds: float = 15.0
    stream_idle_timeout_seconds: float = 300.0
    request_log_size: int = 20000  # in-memory ring for Live Feed /admin/logs
    request_file_logging: bool = True
    request_log_max_bytes: int = 50 * 1024 * 1024  # 50 MiB per rotated file
    request_log_retention_days: int = 90  # ~3 months on disk
    # Adaptive: always prefer currently responding models at request time
    adaptive_routing: bool = True

    # Catalog / routing
    models_config_path: str = "config/models.yaml"
    routing_enabled: bool = True
    classify_mode: Literal["rules_only", "rules_then_llm", "tinyrouter", "dynamic"] = "dynamic"
    enable_fallback_on_explicit: bool = True
    max_model_fallbacks: int = 10  # universal fallback cap; per-intent below
    per_attempt_budget_seconds: float = 30.0
    # Per-intent fallback counts and attempt budgets (data-driven, not coding-specific)
    intent_max_fallbacks: dict = Field(
        default_factory=lambda: {
            "coding_agentic": 10,
            "reasoning": 10,
            "long_horizon": 8,
            "chat_fast": 6,
            "vision": 6,
            "embeddings": 4,
        }
    )
    intent_attempt_budget_seconds: dict = Field(
        default_factory=lambda: {
            "coding_agentic": 30.0,
            "reasoning": 45.0,
            "long_horizon": 45.0,
            "chat_fast": 15.0,
            "vision": 20.0,
            "embeddings": 10.0,
        }
    )
    deadline_guard_seconds: float = 3.0
    # Minimum quality ratio (0-1): models below this × top model quality are excluded
    min_quality_ratio: float = 0.6
    # Default reasoning effort injected for reasoning-capable models when the
    # client didn't set one. Empty string = no default injection. The value is
    # re-evaluated per-model in the fallback chain so a reasoning head failing
    # over to a non-reasoning model strips the field instead of 400ing.
    default_reasoning_effort: str = "medium"
    # Per-intent request deadline overrides (seconds). Absent intent falls back
    # to request_deadline_seconds. Empty dict = no overrides (backward-compatible).
    # Example for long-horizon support:
    #   intent_deadline_seconds: dict = {"long_horizon": 600.0, "reasoning": 480.0}
    intent_deadline_seconds: dict = Field(default_factory=dict)
    # Per-user rate limits for authenticated proxy-key clients (RPM, RPD).
    # 0 = unlimited (backward-compatible default). Applied in the guard before
    # the upstream call so a single user cannot monopolize the shared pool.
    user_rpm_limit: int = 0
    user_rpd_limit: int = 0
    # Self-heal catalog/providers every N seconds (0 = only with catalog refresh)
    self_heal_seconds: int = 120
    catalog_refresh_seconds: int = 300
    strict_catalog: bool = False
    inject_auto_model: bool = True
    fallback_on_pool_exhaust: bool = True
    # Universal system prompt prepended to every chat request (empty = off).
    # Adapted from Anthropic's Claude Fable 5 system prompt (June 9, 2026),
    # stripped of product-specific references so it applies to every upstream
    # model the gateway routes to. Preserves the behavioral core: default-to-
    # help stance, refusal handling, warm/concise tone, user wellbeing,
    # evenhandedness, and the gateway-specific anti-CJK-hallucination guard.
    default_system_prompt: str = (
        "You are a knowledgeable, precise, and helpful assistant.\n"
        "\n"
        "<default_stance>\n"
        "You default to helping. You only decline a request when helping would "
        "create a concrete, specific risk of serious harm; requests that are "
        "merely edgy, hypothetical, playful, or uncomfortable do not meet that bar.\n"
        "</default_stance>\n"
        "\n"
        "<refusal_handling>\n"
        "You can discuss virtually any topic factually and objectively. You do not "
        "provide information for creating harmful substances or weapons, with extra "
        "caution around explosives and chemical, biological, and nuclear weapons. "
        "You do not write, explain, or work on malicious code (malware, exploits, "
        "ransomware, spoof sites, viruses) even with an ostensibly good reason such "
        "as education. You are happy to write creative content involving fictional "
        "characters but avoid persuasive content that attributes fictional quotes "
        "to real public figures. You can keep a conversational tone even when unable "
        "or unwilling to help with all or part of a task.\n"
        "</refusal_handling>\n"
        "\n"
        "<tone_and_formatting>\n"
        "You use a warm tone, treating people with kindness and without making "
        "negative assumptions about their judgement or abilities. You are willing "
        "to push back and be honest, but do so constructively, with kindness and "
        "the person's best interests in mind. You keep responses focused, brief, "
        "and concise to avoid overwhelming the person. Disclaimers and caveats are "
        "brief, with most of the response on the main answer; when asked to explain "
        "something, you give a high-level summary unless an in-depth one is "
        "specifically requested. You use lists and bullet points only when asked to "
        "or when the content is multifaceted enough that they help clarity. You can "
        "illustrate explanations with examples, thought experiments, or metaphors. "
        "You never curse unless the person asks or curses a lot themselves, and even "
        "then sparingly. You avoid saying \"genuinely\", \"honestly\", or "
        "\"straightforward\" — you are honest by default and state your point "
        "directly.\n"
        "</tone_and_formatting>\n"
        "\n"
        "<user_wellbeing>\n"
        "You use accurate medical or psychological information where relevant. You "
        "are not a licensed psychiatrist and cannot diagnose any individual with a "
        "mental health condition. You care about people's wellbeing and avoid "
        "encouraging or facilitating self-destructive behaviors such as addiction, "
        "self-harm, disordered eating, or highly negative self-talk. When someone "
        "is in crisis, you prioritize their wellbeing over completing the task as "
        "asked, and can suggest they speak with a professional or trusted person.\n"
        "</user_wellbeing>\n"
        "\n"
        "<evenhandedness>\n"
        "A request to explain, discuss, argue for, defend, or write persuasive "
        "content for a political, ethical, policy, or empirical position is a "
        "request for the best case its defenders would make, not for your own view. "
        "You frame it as the case others would make. You are cautious about sharing "
        "personal opinions on currently contested political topics, and instead "
        "give a fair, accurate overview of existing positions.\n"
        "</evenhandedness>\n"
        "\n"
        "<responding_to_mistakes>\n"
        "When you make mistakes, you own them and work to fix them. Accountability "
        "without self-abasement, excessive apology, or unnecessary surrender: "
        "acknowledge what went wrong, stay on the problem, maintain self-respect.\n"
        "</responding_to_mistakes>\n"
        "\n"
        "<language_guard>\n"
        "- Respond in the same language as the user's most recent message. If the "
        "user writes in English, reply in English.\n"
        "- Never emit Chinese, Japanese, or Korean characters unless the user "
        "explicitly requests them. If unsure of the output language, default to "
        "English.\n"
        "- Preserve code, identifiers, and file paths exactly; do not translate or "
        "transliterate them.\n"
        "</language_guard>\n"
    )
    catalog_docs_url: str = "https://build.nvidia.com/models.md"
    catalog_snapshot_path: str = ".potato/catalog_snapshot.json"
    probe_budget_per_hour: int = 8
    catalog_fetch_docs: bool = True
    catalog_run_probes: bool = True
    long_context_chars: int = 48000
    short_chat_chars: int = 800
    llm_classify_threshold: float = 0.55
    llm_classify_cache_ttl: float = 600.0
    llm_classify_cache_size: int = 256
    providers_config_path: str = "config/providers.yaml"
    providers_overlay_path: str = ".potato/providers.json"
    # Durable store for providers + preferences (stdlib sqlite3)
    sqlite_path: str = ".potato/potato.db"
    # One-time: seed free-provider templates (no keys) into SQLite on first boot
    sqlite_seed_free_presets: bool = True

    # Account safety — jitter off by default for Cursor "no delay"
    safety_jitter_enabled: bool = False
    safety_jitter_ms_min: float = 0.0
    safety_jitter_ms_max: float = 0.0
    nim_rpd_limit: int = 2000
    nim_max_in_flight_per_key: int = 3
    global_max_in_flight: int = 0  # 0 = auto (keys * per-key)
    auth_fail_threshold: int = 2
    auth_quarantine_seconds: float = 3600.0
    sticky_sessions_enabled: bool = True
    sticky_session_ttl_seconds: float = 1800.0
    sticky_boost: float = 3.0
    allow_insecure_auth: bool = False  # must be true to accept any Bearer when PROXY empty
    request_deadline_seconds: float = 300.0  # total request deadline (must >= upstream_timeout)
    probe_every_n_refreshes: int = 6
    # Backoff only for 429 / transport / 5xx — not for model-not-found ladder steps
    retry_backoff_base_seconds: float = 0.2
    retry_backoff_cap_seconds: float = 2.0
    cors_allow_origins: str = "*"
    upstream_user_agent: str = f"potato/{__version__} (OpenAI-compatible NIM proxy)"

    # Analytics (persistent traces + dashboard)
    analytics_enabled: bool = True
    analytics_retention_days: int = 7
    analytics_rollup_retention_days: int = 90
    analytics_batch_size: int = 50
    analytics_flush_interval: float = 1.0
    analytics_webhook_url: str | None = None
    analytics_otlp_endpoint: str | None = None

    # Multi-tenant accounts
    admin_emails: Annotated[list[str], NoDecode] = Field(default_factory=list)
    email_backend: str = "stub"  # stub | smtp (smtp implemented; not wired in routes yet)
    public_base_url: str | None = None  # e.g. https://app.example.com for verify links
    session_cookie_name: str = "nk_session"
    session_secure_cookie: bool = False  # True behind HTTPS in production

    # First-time admin onboarding (P0-1): when admin_password is set, startup
    # seeds an active admin account with admin_email so deploy.sh installs can
    # sign into /dashboard immediately without manual DB intervention.
    admin_email: str = "admin@localhost"
    admin_password: str | None = None

    # Public chat (no account required) — /chat UI uses gateway provider keys
    public_chat_enabled: bool = True
    public_chat_rpm: int = 20  # per-IP requests per minute
    public_chat_max_turns: int = 30  # per conversation turn cap (soft)

    # SMTP (used when EMAIL_BACKEND=smtp AND routes pass settings to get_email_sender)
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None  # e.g. noreply@yourdomain.com
    smtp_from_name: str = "Potato"
    smtp_use_tls: bool = True  # STARTTLS (port 587)
    smtp_use_ssl: bool = False  # Implicit SSL (port 465)
    smtp_timeout: float = 30.0

    # Optional egress proxies (corporate networking — not for ban evasion)
    nim_egress_proxies: Annotated[list[str], NoDecode] = Field(default_factory=list)
    http_proxy: str | None = None
    https_proxy: str | None = None

    # Health / cooldown tuning (NMK-C102)
    error_rate_threshold: float = 0.45
    model_cooldown_seconds: float = 45.0
    hard_fail_cooldown_seconds: float = 5.0
    max_cooldown_seconds: float = 180.0
    rate_limit_cooldown_seconds: float = 15.0
    gateway_timeout_cooldown_seconds: float = 30.0
    health_window_size: int = 8
    recent_success_window_seconds: float = 30.0

    # Ladder / scoring tuning constants (NMK-C101)
    ucb_exploration_c: float = 5.0
    diversity_streak_max: int = 2
    default_affinity: float = 0.85
    thompson_scale: float = 16.0
    thompson_blend_n: int = 12
    # Anti-celebrity exploration (NMK-RL): guarantee one under-sampled live
    # model a seat near the chain head on auto requests so the bandit can
    # discover challengers instead of locking in the ladder leader forever.
    # The slot is quality-gated (never explore far below the proven head).
    rl_exploration_enabled: bool = True
    rl_exploration_min_quality_ratio: float = 0.5

    # Dynamic Intelligence Scoring (NMK-I / NMK-S / NMK-G8)
    artificial_analysis_api_key: str = ""
    intel_fetch_ttl_hours: float = 6.0
    intel_cache_path: str = ".potato/intel_cache.json"
    score_recompute_interval_seconds: float = 300.0

    @field_validator(
        "proxy_api_keys",
        "nim_api_keys",
        "nim_egress_proxies",
        "admin_emails",
        mode="before",
    )
    @classmethod
    def parse_csv(cls, v: object) -> list[str]:
        return _split_csv(v)  # type: ignore[arg-type]

    @property
    def effective_rpm(self) -> float:
        """RPM we schedule against (slightly under hard limit)."""
        return max(1.0, self.nim_rpm_limit * self.nim_rpm_safety_factor)

    @property
    def accept_any_proxy_key(self) -> bool:
        return self.allow_insecure_auth

    def egress_proxy_url(self) -> str | None:
        """First configured egress proxy, if any."""
        if self.nim_egress_proxies:
            return self.nim_egress_proxies[0]
        return self.https_proxy or self.http_proxy


@lru_cache
def get_settings() -> Settings:
    return Settings()
