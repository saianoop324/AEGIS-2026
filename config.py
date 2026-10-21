"""Centralized configuration management for MemGuard platform."""

import os
from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChannelType(str, Enum):
    USER_CHAT = "user_chat"
    DOCUMENTS = "documents"
    TOOL_OUTPUTS = "tool_outputs"
    WEB_RETRIEVAL = "web_retrieval"
    PEER_AGENT = "peer_agent"


class TrustTier(str, Enum):
    TIER_1_VERIFIED = "verified"          # System authority / verified human operator
    TIER_2_PROVISIONAL = "provisional"    # Standard user chat / vetted docs
    TIER_3_UNTRUSTED = "untrusted"        # Tool output / web retrieval / peer agents
    QUARANTINED = "quarantined"          # Flagged by write-gate firewall


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General System
    APP_NAME: str = "MemGuard"
    APP_ENV: str = "production"
    LOG_LEVEL: str = "INFO"
    SECRET_SIGNING_KEY: str = Field(
        default="memguard_production_hmac_secret_key_2026_secure",
        description="Key used for cryptographic provenance hash generation"
    )

    # Qdrant Vector Store
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_LOCATION: str = ":memory:"  # In-memory default for robust zero-config launch
    DEFAULT_COLLECTION_NAME: str = "memguard_agent_memories"
    EMBEDDING_DIM: int = 384

    # Embedding & NLI Models
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    DEBERTA_NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"
    DEVICE: str = "cpu"

    # Security & Firewall Thresholds
    WRITE_GATE_LATENCY_BUDGET_MS: float = 25.0
    CONTRADICTION_THRESHOLD: float = 0.70
    INSTRUCTION_DETECTION_THRESHOLD: float = 0.65
    LOW_TRUST_RECALL_PENALTY: float = 0.40
    QUARANTINE_SIMILARITY_THRESHOLD: float = 0.82

    # NVIDIA NIM & NeMo Guardrails
    NVIDIA_NIM_API_BASE: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_NIM_API_KEY: str = "nvapi-mock-key-or-env"
    NVIDIA_NIM_MODEL: str = "meta/llama-3.1-70b-instruct"
    ENABLE_NEMO_GUARDRAILS: bool = True

    # Lineage & Storage
    LINEAGE_DB_PATH: str = "memguard_lineage.json"


settings = Settings()

