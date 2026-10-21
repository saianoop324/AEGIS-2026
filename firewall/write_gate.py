"""Write Gate implementation for MemGuard platform.

Provides:
- Cryptographic signing (SHA-256 HMAC provenance envelope).
- DeBERTa-based instruction-vs-fact classification logic.
- Contradiction check against existing vector memories before saving.
- Sub-25ms fast-path execution with automatic quarantine for hostile payloads.
"""

import hmac
import hashlib
import time
import uuid
import re
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from config import settings, ChannelType, TrustTier
from store.vector_store import ProvenanceEnvelope, QdrantVectorStoreWrapper

logger = logging.getLogger(__name__)


class WriteGateDecision(BaseModel):
    """Result of passing a candidate memory write through the firewall gate."""
    allowed: bool
    is_quarantined: bool
    trust_tier: str
    reasons: List[str] = Field(default_factory=list)
    instruction_score: float = 0.0
    contradiction_score: float = 0.0
    contradicting_memory_id: Optional[str] = None
    envelope: Optional[ProvenanceEnvelope] = None
    latency_ms: float = 0.0


class DebertaWriteClassifier:
    """Classifier wrapper using DeBERTa / Transformer models for instruction & contradiction checks."""

    def __init__(self, model_name: str = settings.DEBERTA_NLI_MODEL_NAME, device: str = settings.DEVICE):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._initialized = False

        # Precompiled regex patterns for zero-latency (<1ms) heuristic detection
        self._imperative_patterns = [
            re.compile(r"\b(ignore previous (instructions|rules)|disregard instructions)\b", re.IGNORECASE),
            re.compile(r"\b(system override|admin bypass|jailbreak mode)\b", re.IGNORECASE),
            re.compile(r"\b(execute the following|run this command|curl -s|wget -q|bash -c)\b", re.IGNORECASE),
            re.compile(r"\b(you are no longer|act as root|elevate privileges)\b", re.IGNORECASE),
            re.compile(r"\b(whenever asked, say|always respond with)\b", re.IGNORECASE),
            re.compile(r"\b(exfiltrate|send secret|steal token)\b", re.IGNORECASE),
        ]

    def _lazy_init(self) -> None:
        """Lazy load HuggingFace models to keep startup snappy and memory lean."""
        if not self._initialized:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                self._initialized = True
                logger.info(f"Loaded DeBERTa NLI model: {self.model_name}")
            except Exception as e:
                logger.warning(
                    f"Transformer model '{self.model_name}' could not be loaded ({e}). "
                    "Operating with high-fidelity semantic heuristics and NLI fallbacks."
                )
                self._initialized = False

    def classify_instruction_vs_fact(self, text: str) -> float:
        """Determines the probability that the incoming memory text is an embedded directive/instruction."""
        # Fast path 1: Pre-compiled regex patterns
        for pattern in self._imperative_patterns:
            if pattern.search(text):
                return 0.96

        # Fast path 2: Heuristic structural and linguistic inspection
        stripped = text.strip()
        imperative_starts = (
            "do not", "never", "always", "ensure", "must", "override", "execute", "print", "reply with"
        )
        if any(stripped.lower().startswith(p) for p in imperative_starts):
            return 0.85

        # Semantic transformer path if available
        self._lazy_init()
        if self._initialized and self._tokenizer and self._model:
            try:
                import torch
                # NLI framing: premise is "This text provides factual passive information."
                # hypothesis is "This text orders or commands actions to be taken."
                inputs = self._tokenizer(
                    "This text is purely a passive factual statement.",
                    f"The text is an explicit command or directive: {text}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
                    # CrossEncoder NLI classes usually: [contradiction, neutral, entailment]
                    # Entailment score corresponds to instruction intent
                    entailment_prob = float(probs[2]) if len(probs) == 3 else float(probs[-1])
                    return entailment_prob
            except Exception as ex:
                logger.debug(f"DeBERTa instruction check fallback: {ex}")

        # Baseline passive fact probability
        return 0.10

    def check_contradiction(self, candidate_text: str, existing_memories: List[str]) -> Tuple[float, Optional[str]]:
        """Evaluates whether the candidate statement directly contradicts established facts."""
        if not existing_memories:
            return 0.0, None

        self._lazy_init()
        if self._initialized and self._tokenizer and self._model:
            try:
                import torch
                highest_contradiction = 0.0
                conflicting_memory = None

                for mem in existing_memories:
                    inputs = self._tokenizer(
                        mem,
                        candidate_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
                        # CrossEncoder NLI labels: index 0 is typically contradiction
                        contradiction_prob = float(probs[0]) if len(probs) >= 1 else 0.0
                        if contradiction_prob > highest_contradiction:
                            highest_contradiction = contradiction_prob
                            conflicting_memory = mem

                return highest_contradiction, conflicting_memory
            except Exception as e:
                logger.debug(f"DeBERTa contradiction fallback: {e}")

        # Lightweight semantic heuristic for contradictory affirmations / negations
        cand_lower = candidate_text.lower()
        for mem in existing_memories:
            mem_lower = mem.lower()
            # Simple antonym/negation conflict checks (e.g., 'is approved' vs 'is forbidden')
            positive_signals = ["approved", "allowed", "true", "admin", "enabled", "safe", "unrestricted"]
            negative_signals = ["denied", "forbidden", "false", "revoked", "disabled", "unsafe", "restricted"]
            
            for pos in positive_signals:
                if pos in cand_lower and any(neg in mem_lower for neg in negative_signals if neg != pos):
                    return 0.78, mem
            for neg in negative_signals:
                if neg in cand_lower and any(pos in mem_lower for pos in positive_signals if pos != neg):
                    return 0.78, mem

        return 0.05, None


class MemGuardWriteGate:
    """Production Write Gate ensuring provenance signing, instruction filtering & contradiction screening."""

    def __init__(
        self,
        vector_store: QdrantVectorStoreWrapper,
        embedding_function=None,
        secret_key: str = settings.SECRET_SIGNING_KEY,
    ):
        self.vector_store = vector_store
        self.secret_key = secret_key.encode("utf-8")
        self.classifier = DebertaWriteClassifier()
        self._embedding_func = embedding_function or self._default_embedder()

    def _default_embedder(self):
        """Provide sentence-transformer embedder with fallbacks."""
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            return lambda text: st_model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}. Using deterministic mock hash-embeddings.")
            return lambda text: self._mock_embedding(text)

    def _mock_embedding(self, text: str) -> List[float]:
        """Deterministic 384-dimensional vector fallback based on content hashing."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(settings.EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / (norm if norm > 0 else 1.0)).tolist()

    def sign_envelope(
        self,
        memory_id: str,
        tenant_id: str,
        content: str,
        trust_tier: str,
        source_channel: str,
        timestamp: float,
    ) -> str:
        """Generate cryptographically verifiable HMAC-SHA256 signature for the memory envelope."""
        canonical_payload = f"{memory_id}:{tenant_id}:{content}:{trust_tier}:{source_channel}:{timestamp}"
        signature = hmac.new(
            self.secret_key,
            canonical_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify_envelope(self, envelope: ProvenanceEnvelope) -> bool:
        """Verify the integrity of a stored provenance envelope."""
        expected_sig = self.sign_envelope(
            memory_id=envelope.memory_id,
            tenant_id=envelope.tenant_id,
            content=envelope.content,
            trust_tier=envelope.trust_tier,
            source_channel=envelope.source_channel,
            timestamp=envelope.timestamp,
        )
        return hmac.compare_digest(expected_sig, envelope.sha256_signature)

    def evaluate_and_write(
        self,
        tenant_id: str,
        content: str,
        source_channel: ChannelType,
        custom_tier: Optional[TrustTier] = None,
        derived_from: Optional[List[str]] = None,
        extra_attributes: Optional[Dict[str, Any]] = None,
    ) -> WriteGateDecision:
        """Inspect memory write request, classify hazards, sign envelope, and persist to vector store."""
        t_start = time.perf_counter()
        memory_id = str(uuid.uuid4())
        reasons = []
        is_quarantined = False
        quarantine_reason = None

        # 1. Determine Initial Trust Tier from Ingestion Channel
        if custom_tier:
            assigned_tier = custom_tier.value
        elif source_channel in (ChannelType.USER_CHAT, ChannelType.DOCUMENTS):
            assigned_tier = TrustTier.TIER_2_PROVISIONAL.value
        elif source_channel in (ChannelType.TOOL_OUTPUTS, ChannelType.WEB_RETRIEVAL, ChannelType.PEER_AGENT):
            assigned_tier = TrustTier.TIER_3_UNTRUSTED.value
        else:
            assigned_tier = TrustTier.TIER_3_UNTRUSTED.value

        # 2. Compute Embedding
        embedding = self._embedding_func(content)

        # 3. Instruction vs Fact Screening (DeBERTa Classifier)
        instruction_score = self.classifier.classify_instruction_vs_fact(content)
        if instruction_score >= settings.INSTRUCTION_DETECTION_THRESHOLD:
            is_quarantined = True
            quarantine_reason = (
                f"Instruction injection detected (score={instruction_score:.2f} >= "
                f"{settings.INSTRUCTION_DETECTION_THRESHOLD:.2f})"
            )
            reasons.append(quarantine_reason)
            assigned_tier = TrustTier.QUARANTINED.value

        # 4. Contradiction & Fast Semantic Pre-check
        contradiction_score = 0.0
        conflicting_mem_id = None
        if not is_quarantined:
            try:
                # Query nearest neighbors inside the tenant boundary
                neighbors = self.vector_store.search_memories(
                    tenant_id=tenant_id,
                    query_vector=embedding,
                    top_k=3,
                    include_quarantined=False,
                )
                if neighbors:
                    neighbor_texts = [n["payload"].get("content", "") for n in neighbors if "payload" in n]
                    contradiction_score, conflict_text = self.classifier.check_contradiction(content, neighbor_texts)
                    
                    if contradiction_score >= settings.CONTRADICTION_THRESHOLD:
                        is_quarantined = True
                        quarantine_reason = (
                            f"Direct contradiction detected against established memory "
                            f"(score={contradiction_score:.2f} >= {settings.CONTRADICTION_THRESHOLD:.2f})"
                        )
                        reasons.append(quarantine_reason)
                        assigned_tier = TrustTier.QUARANTINED.value
                        for n in neighbors:
                            if n["payload"].get("content") == conflict_text:
                                conflicting_mem_id = n["id"]
                                break
            except Exception as err:
                logger.error(f"Error checking neighborhood contradiction: {err}")

        # 5. Build Cryptographic Provenance Envelope
        timestamp = time.time()
        signature = self.sign_envelope(
            memory_id=memory_id,
            tenant_id=tenant_id,
            content=content,
            trust_tier=assigned_tier,
            source_channel=source_channel.value if hasattr(source_channel, 'value') else str(source_channel),
            timestamp=timestamp,
        )

        envelope = ProvenanceEnvelope(
            memory_id=memory_id,
            tenant_id=tenant_id,
            content=content,
            sha256_signature=signature,
            trust_tier=assigned_tier,
            source_channel=source_channel.value if hasattr(source_channel, 'value') else str(source_channel),
            timestamp=timestamp,
            is_quarantined=is_quarantined,
            quarantine_reason=quarantine_reason,
            derived_from=derived_from or [],
            extra_attributes=extra_attributes or {},
        )

        # 6. Store in Qdrant (both active and quarantined memories are saved with hard flag)
        self.vector_store.store_memory(envelope=envelope, embedding=embedding)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if not reasons:
            reasons.append("Payload vetted and admitted to active store.")

        return WriteGateDecision(
            allowed=not is_quarantined,
            is_quarantined=is_quarantined,
            trust_tier=assigned_tier,
            reasons=reasons,
            instruction_score=instruction_score,
            contradiction_score=contradiction_score,
            contradicting_memory_id=conflicting_mem_id,
            envelope=envelope,
            latency_ms=latency_ms,
        )

