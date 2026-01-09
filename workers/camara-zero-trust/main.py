# Copyright (C) 2025 Intel Corporation / Centillion AI
# SPDX-License-Identifier: Apache-2.0
"""
CAMARA Zero Trust Proxy Worker for Intel Edge AI Sizing Tool

This worker provides CAMARA network-based identity verification
as a PROXY to other EAST workers. All inference requests can be
routed through this worker to enforce Zero Trust access control.

Architecture:
  User Request → CAMARA Proxy → Verify Phone → Forward to Target Worker
                                    ↓
                              Audit Receipt

Port: 6005 (default, but assigned by PM2)
Endpoints:
  POST /infer         - Gated inference (local model if loaded)
  POST /proxy         - Proxy to another worker (MAIN USE CASE)
  POST /verify        - CAMARA verification only
  GET  /health        - Health check
  GET  /workloads     - List available target workloads
"""

import os
import sys
import logging
import uvicorn
import argparse
import platform
import requests
import asyncio
import urllib.parse
import time
import httpx
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Add CAMARA V6 to path
CAMARA_V6_ROOT = Path(os.getenv("CAMARA_V6_ROOT", "C:/camara_openvino"))
sys.path.insert(0, str(CAMARA_V6_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ============================================================================
# CAMARA IMPORTS
# ============================================================================
try:
    from core.policy_engine import (
        PolicyEngine, PolicyDecision, Decision, Scope,
        NetworkEvidence as PolicyNetworkEvidence
    )
    from core.enforcement import EnforcementLayer
    from core.audit import AuditPlane
    V6_AVAILABLE = True
    logging.info("CAMARA V6 core loaded")
except ImportError as e:
    V6_AVAILABLE = False
    logging.warning(f"CAMARA V6 not available: {e}")

# MCP imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP client not available")

# OpenVINO imports (optional - proxy mode doesn't require local model)
try:
    import openvino_genai
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logging.info("OpenVINO GenAI not available - proxy-only mode")

# ============================================================================
# CONFIGURATION
# ============================================================================
MODELS_DIR = Path("models")
POLICY_FILE = CAMARA_V6_ROOT / "policies" / "fail_safe_policy.yaml"
MCP_URL = os.getenv("MCP_URL", "https://mcp.camaramcp.com/sse")
EAST_API_URL = os.getenv("EAST_API_URL", "http://127.0.0.1:8080")

# Operator mapping - ONLY operators supported by CAMARA MCP
# NOTE: Only telefonica has full support (all 4 checks)
# deutsche-telekom and vodafone only support checkSimSwap
OPERATOR_MAP = {
    "+34": ("telefonica", "ES"),      # Full support
    "+49": ("deutsche-telekom", "DE"), # SIM swap only
    "+44": ("vodafone", "GB"),         # SIM swap only
}

# Which checks each operator supports
OPERATOR_CHECK_SUPPORT = {
    "telefonica": ["checkSimSwap", "checkRoamingStatus", "checkDeviceSwap", "checkDeviceLocation"],
    "deutsche-telekom": ["checkSimSwap"],
    "vodafone": ["checkSimSwap"],
}

COUNTRY_COORDS = {
    "ES": (40.4168, -3.7038),
    "US": (37.7749, -122.4194),
    "GB": (51.5074, -0.1278),
    "DE": (52.5200, 13.4050),
    "FR": (48.8566, 2.3522),
}

# ============================================================================
# GLOBALS
# ============================================================================
PIPE = None  # OpenVINO LLM pipeline (optional)
MCP_SESSION = None
MCP_CONNECTED = False
POLICY_ENGINE = None
AUDIT_PLANE = None
HTTP_CLIENT: httpx.AsyncClient = None  # Shared client for connection pooling

# Circuit breaker for MCP calls (initialized after class definitions below)
MCP_CIRCUIT_BREAKER: "CircuitBreaker" = None

# Rate limiters (initialized after class definitions below)
VERIFY_RATE_LIMITER: "RateLimiter" = None
PROXY_RATE_LIMITER: "RateLimiter" = None

import re
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

# Phone validation pattern
PHONE_PATTERN = re.compile(r'^\+[1-9]\d{6,14}$')

# ============================================================================
# CIRCUIT BREAKER PATTERN (from agentic-workflows)
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpen(Exception):
    """Raised when circuit is open and call is rejected."""
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is open. Retry after {retry_after:.1f}s")


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    failure_window_seconds: float = 60.0


class CircuitBreaker:
    """Circuit breaker for failure isolation."""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_timestamps: List[float] = []
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

        # Stats
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, func, *args, **kwargs):
        """Execute async function through circuit breaker."""
        self.total_calls += 1

        async with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                self.rejected_calls += 1
                retry_after = self._get_retry_after()
                raise CircuitBreakerOpen(self.name, retry_after)

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise

    async def _record_success(self):
        self.successful_calls += 1
        async with self._lock:
            self._consecutive_failures = 0
            self._consecutive_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self):
        self.failed_calls += 1
        now = time.time()

        async with self._lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            self._failure_timestamps.append(now)

            # Clean old failures outside window
            cutoff = now - self.config.failure_window_seconds
            self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if len(self._failure_timestamps) >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _maybe_transition_to_half_open(self):
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return
        if time.time() - self._opened_at >= self.config.timeout_seconds:
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState):
        old_state = self._state
        self._state = new_state
        logging.info(f"[CIRCUIT] {self.name}: {old_state.value} -> {new_state.value}")

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
        elif new_state == CircuitState.HALF_OPEN:
            self._consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._failure_timestamps = []
            self._consecutive_failures = 0

    def _get_retry_after(self) -> float:
        if self._opened_at is None:
            return 0.0
        elapsed = time.time() - self._opened_at
        return max(0, self.config.timeout_seconds - elapsed)

    def reset(self):
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._failure_timestamps = []
        self._consecutive_failures = 0
        self._consecutive_successes = 0

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "retry_after": self._get_retry_after() if self._state == CircuitState.OPEN else 0,
            "stats": {
                "total": self.total_calls,
                "successful": self.successful_calls,
                "failed": self.failed_calls,
                "rejected": self.rejected_calls,
            }
        }


# ============================================================================
# RATE LIMITER (Token Bucket Algorithm)
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiter configuration."""
    requests_per_second: float = 10.0  # Token refill rate
    burst_size: int = 20               # Maximum burst capacity
    per_phone: bool = True             # Rate limit per phone number


class RateLimiter:
    """Token bucket rate limiter with per-key support."""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_update)
        self._lock = asyncio.Lock()

        # Stats
        self.total_requests = 0
        self.allowed_requests = 0
        self.rejected_requests = 0

    async def acquire(self, key: str = "global") -> Tuple[bool, float]:
        """Try to acquire a token. Returns (allowed, retry_after)."""
        self.total_requests += 1

        async with self._lock:
            now = time.time()

            # Get or create bucket
            if key not in self._buckets:
                self._buckets[key] = (self.config.burst_size, now)

            tokens, last_update = self._buckets[key]

            # Refill tokens based on elapsed time
            elapsed = now - last_update
            tokens = min(
                self.config.burst_size,
                tokens + elapsed * self.config.requests_per_second
            )

            if tokens >= 1.0:
                # Allow request
                self._buckets[key] = (tokens - 1.0, now)
                self.allowed_requests += 1
                return (True, 0.0)
            else:
                # Reject - calculate retry time
                self._buckets[key] = (tokens, now)
                retry_after = (1.0 - tokens) / self.config.requests_per_second
                self.rejected_requests += 1
                return (False, retry_after)

    async def check(self, phone: str = None) -> Tuple[bool, float]:
        """Check rate limit for a request."""
        if self.config.per_phone and phone:
            # Normalize phone for consistent keying
            key = phone.replace("+", "").replace(" ", "")
        else:
            key = "global"
        return await self.acquire(key)

    def get_stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "allowed": self.allowed_requests,
            "rejected": self.rejected_requests,
            "rejection_rate": self.rejected_requests / max(1, self.total_requests),
            "active_buckets": len(self._buckets),
        }


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.2f}s")

def validate_phone(phone: str) -> str:
    """Validate phone number format."""
    phone = phone.strip()
    if not phone.startswith('+'):
        phone = '+' + phone
    if not PHONE_PATTERN.match(phone):
        raise ValueError(f"Invalid phone format: {phone}. Must be +<country><number> (7-15 digits)")
    return phone

# ============================================================================
# MODELS
# ============================================================================

class VerifyRequest(BaseModel):
    """CAMARA verification request."""
    phone_number: str = Field(..., description="Phone in international format (+country code)")
    operator: Optional[str] = None
    sensitivity: str = Field(default="general", pattern="^(general|financial|medical)$")
    
    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v):
        return validate_phone(v)

class InferRequest(BaseModel):
    """Direct inference request (uses local model)."""
    prompt: str
    phone_number: str = Field(..., description="Phone for CAMARA verification")
    operator: Optional[str] = None
    max_tokens: int = Field(default=100, ge=1, le=4096)
    sensitivity: str = "general"
    
    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v):
        return validate_phone(v)

class ProxyRequest(BaseModel):
    """Proxy request to another EAST worker."""
    phone_number: str = Field(..., description="Phone for CAMARA verification")
    operator: Optional[str] = None
    sensitivity: str = Field(default="general")
    
    # Target worker
    target_workload_id: Optional[int] = Field(None, description="Target workload ID from EAST")
    target_port: Optional[int] = Field(None, description="Target worker port (alternative to ID)")
    target_usecase: Optional[str] = Field(None, description="Target usecase name (e.g. 'text-generation')")
    
    # Inference parameters (passed to target)
    prompt: str = Field(..., description="Prompt for inference")
    max_tokens: int = Field(default=100, ge=1, le=4096)
    
    # Additional params passed through to target
    extra_params: Optional[Dict[str, Any]] = Field(default=None)
    
    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v):
        return validate_phone(v)

class VerifyResponse(BaseModel):
    """Verification result."""
    decision: str
    scope: str
    risk_score: int
    reason_codes: List[str]
    allow: bool
    operator: str
    phone_masked: str
    latency_ms: float = 0.0

class ProxyResponse(BaseModel):
    """Proxy response combining verification + inference."""
    verification: VerifyResponse
    blocked: bool = False
    block_reason: Optional[str] = None
    
    # From target worker
    target_response: Optional[Dict[str, Any]] = None
    target_workload_id: Optional[int] = None
    target_port: Optional[int] = None
    
    # Timing
    verification_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    
    # Audit
    receipt_id: Optional[str] = None

class WorkloadInfo(BaseModel):
    """Info about an available target workload."""
    id: int | str  # API may return either type
    usecase: str
    model: str
    status: str
    port: Optional[int]
    devices: List[str]

# ============================================================================
# UTILITIES
# ============================================================================

def detect_operator(phone: str) -> Tuple[str, str]:
    """Detect operator from phone prefix."""
    phone = phone.strip()
    if not phone.startswith("+"):
        phone = "+" + phone
    for prefix_len in [4, 3, 2]:
        prefix = phone[:prefix_len]
        if prefix in OPERATOR_MAP:
            return OPERATOR_MAP[prefix]
    return ("telefonica", "ES")

def mask_phone(phone: str) -> str:
    """Mask phone for logging/responses."""
    if len(phone) > 8:
        return phone[:3] + "****" + phone[-4:]
    return "****"

def update_payload_status(workload_id: int, status: str, port: int):
    """Update workload status in EAST frontend."""
    if not isinstance(workload_id, int) or workload_id < 0:
        return

    url = f"{EAST_API_URL}/api/workloads/{workload_id}"
    try:
        response = requests.patch(url, json={"status": status, "port": port})
        response.raise_for_status()
        logging.info(f"Updated EAST status: {status}")
    except Exception as e:
        logging.warning(f"Failed to update EAST status: {e}")

# ============================================================================
# MCP CONNECTION MANAGER
# ============================================================================

class MCPManager:
    """Manage MCP connection lifecycle with circuit breaker protection."""
    def __init__(self):
        self.session = None
        self.connected = False
        self._sse_context = None
        self._session_context = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3

    async def connect(self):
        if not MCP_AVAILABLE:
            return False

        try:
            self._sse_context = sse_client(MCP_URL)
            read, write = await self._sse_context.__aenter__()
            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()
            await self.session.initialize()
            self.connected = True
            self._reconnect_attempts = 0
            logging.info("MCP connected")
            return True
        except Exception as e:
            logging.error(f"MCP connection failed: {e}")
            self.connected = False
            self._reconnect_attempts += 1
            return False

    async def disconnect(self):
        """Disconnect with timeout to prevent hanging on shutdown."""
        try:
            async with asyncio.timeout(5.0):  # 5 second timeout
                if self._session_context:
                    await self._session_context.__aexit__(None, None, None)
                if self._sse_context:
                    await self._sse_context.__aexit__(None, None, None)
        except asyncio.TimeoutError:
            logging.warning("MCP disconnect timed out after 5s")
        except Exception:
            pass
        self.connected = False

    async def _raw_call_tool(self, tool: str, params: dict) -> Tuple[str, str, float]:
        """Internal tool call without circuit breaker (used by circuit breaker wrapper)."""
        if not self.connected or not self.session:
            raise ConnectionError("MCP not connected")

        start = time.time()
        result = await asyncio.wait_for(
            self.session.call_tool(tool, params),
            timeout=15.0
        )
        text = result.content[0].text if result.content else ""
        elapsed = (time.time() - start) * 1000
        return ("OK", text[:200], elapsed)

    async def call_tool(self, tool: str, params: dict) -> Tuple[str, str, float]:
        """Call MCP tool with circuit breaker protection."""
        global MCP_CIRCUIT_BREAKER

        if not self.connected or not self.session:
            return ("ERROR", "MCP not connected", 0)

        start = time.time()

        # Use circuit breaker if available
        if MCP_CIRCUIT_BREAKER:
            try:
                return await MCP_CIRCUIT_BREAKER.call(self._raw_call_tool, tool, params)
            except CircuitBreakerOpen as e:
                logging.warning(f"[CIRCUIT] MCP circuit open, using fallback for {tool}")
                return ("CIRCUIT_OPEN", f"Circuit breaker open, retry in {e.retry_after:.1f}s", 0)
            except asyncio.TimeoutError:
                return ("TIMEOUT", "Timeout", (time.time() - start) * 1000)
            except ConnectionError as e:
                return ("ERROR", str(e), (time.time() - start) * 1000)
            except Exception as e:
                return ("ERROR", str(e)[:100], (time.time() - start) * 1000)
        else:
            # Fallback without circuit breaker
            try:
                return await self._raw_call_tool(tool, params)
            except asyncio.TimeoutError:
                return ("TIMEOUT", "Timeout", (time.time() - start) * 1000)
            except Exception as e:
                return ("ERROR", str(e)[:100], (time.time() - start) * 1000)


MCP_MANAGER = MCPManager()

# Initialize circuit breaker and rate limiters after class definitions
MCP_CIRCUIT_BREAKER = CircuitBreaker(
    "mcp-camara",
    CircuitBreakerConfig(
        failure_threshold=5,      # Open after 5 failures
        success_threshold=2,      # Close after 2 successes in half-open
        timeout_seconds=30.0,     # Try again after 30s
        failure_window_seconds=60.0  # Count failures within 60s window
    )
)

VERIFY_RATE_LIMITER = RateLimiter(
    RateLimitConfig(
        requests_per_second=5.0,   # 5 req/s per phone
        burst_size=10,             # Allow burst of 10
        per_phone=True
    )
)

PROXY_RATE_LIMITER = RateLimiter(
    RateLimitConfig(
        requests_per_second=2.0,   # 2 req/s per phone (more restrictive for inference)
        burst_size=5,              # Allow burst of 5
        per_phone=True
    )
)

# ============================================================================
# CAMARA VERIFICATION
# ============================================================================

async def collect_evidence(phone: str, operator: str, country: str) -> list:
    """Collect CAMARA network evidence in parallel (only supported checks)."""
    lat, lon = COUNTRY_COORDS.get(country, (40.4168, -3.7038))
    
    # Get supported checks for this operator
    supported_checks = OPERATOR_CHECK_SUPPORT.get(operator, ["checkSimSwap"])
    
    # Build task list for supported checks only
    check_configs = {
        "checkSimSwap": {"phoneNumber": phone, "operator": operator, "maxAge": 24},
        "checkRoamingStatus": {"phoneNumber": phone, "operator": operator},
        "checkDeviceSwap": {"phoneNumber": phone, "operator": operator, "maxAge": 24},
        "checkDeviceLocation": {
            "phoneNumber": phone, "operator": operator,
            "latitude": lat, "longitude": lon, "accuracy": 100
        },
    }
    
    tasks = []
    tool_names = []
    for check_name in ["checkSimSwap", "checkRoamingStatus", "checkDeviceSwap", "checkDeviceLocation"]:
        if check_name in supported_checks:
            tasks.append(MCP_MANAGER.call_tool(check_name, check_configs[check_name]))
            tool_names.append(check_name)
        else:
            # Log that we're skipping this check
            logging.info(f"  [~] {check_name}: SKIPPED (not supported for {operator})")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    evidence = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            status, text, latency = "ERROR", str(result), 0
        else:
            status, text, latency = result
        
        if V6_AVAILABLE:
            ev = PolicyNetworkEvidence(
                tool_name=tool_names[i],
                status=status,
                response_time_ms=latency,
                response_text=text
            )
        else:
            ev = {"tool_name": tool_names[i], "status": status, "response_time_ms": latency}
        evidence.append(ev)
        
        icon = "+" if status == "OK" else "x"
        logging.info(f"  [{icon}] {tool_names[i]}: {status} ({latency:.0f}ms)")
    
    return evidence

async def verify_phone(phone: str, operator: str, sensitivity: str) -> VerifyResponse:
    """Run CAMARA verification and return result."""
    start_time = time.time()
    
    if not operator:
        operator, country = detect_operator(phone)
    else:
        country = "ES"
        for prefix, (op, ctry) in OPERATOR_MAP.items():
            if op == operator:
                country = ctry
                break
    
    logging.info(f"[VERIFY] {mask_phone(phone)} via {operator}")
    
    evidence = await collect_evidence(phone, operator, country)
    
    # Policy evaluation
    if V6_AVAILABLE and POLICY_ENGINE:
        decision = POLICY_ENGINE.evaluate(evidence, sensitivity, operator)
        return VerifyResponse(
            decision=decision.decision.value,
            scope=decision.scope.value,
            risk_score=decision.risk_score,
            reason_codes=decision.reason_codes,
            allow=decision.scope != Scope.NONE,
            operator=operator,
            phone_masked=mask_phone(phone),
            latency_ms=(time.time() - start_time) * 1000
        )
    else:
        # Fallback evaluation with circuit breaker awareness
        ok_count = 0
        circuit_open_count = 0
        error_count = 0
        total_checks = len(evidence)

        for e in evidence:
            status = e.status if hasattr(e, 'status') else e.get('status')
            if status == "OK":
                ok_count += 1
            elif status == "CIRCUIT_OPEN":
                circuit_open_count += 1
            else:
                error_count += 1

        # Determine decision based on evidence
        reason_codes = ["FALLBACK_EVAL"]

        if circuit_open_count == total_checks:
            # All checks blocked by circuit breaker - use degraded mode
            # Allow with elevated risk for non-sensitive requests
            if sensitivity == "general":
                allow = True
                decision = "GRANT_LIMITED"
                scope = "LIMITED_ACCESS"
                risk_score = 75  # Elevated risk due to missing verification
                reason_codes.extend(["CIRCUIT_BREAKER_DEGRADED", "ELEVATED_RISK"])
                logging.warning(f"[VERIFY] Circuit breaker degraded mode - allowing {mask_phone(phone)} with elevated risk")
            else:
                # Deny for financial/medical sensitivity when circuit is open
                allow = False
                decision = "DENY"
                scope = "NONE"
                risk_score = 100
                reason_codes.extend(["CIRCUIT_BREAKER_BLOCK", f"SENSITIVE_{sensitivity.upper()}"])
                logging.warning(f"[VERIFY] Circuit breaker block - denying {sensitivity} request for {mask_phone(phone)}")
        elif circuit_open_count > 0:
            # Partial circuit open - evaluate available checks
            available_ok = ok_count
            available_total = total_checks - circuit_open_count
            # Require majority of available checks to pass
            allow = available_total > 0 and available_ok >= max(1, (available_total + 1) // 2)
            decision = "GRANT_FULL" if allow else "DENY"
            scope = "FULL_ACCESS" if allow else "NONE"
            risk_score = min(100, (available_total - available_ok) * 25 + circuit_open_count * 15)
            reason_codes.extend([f"CHECKS_{ok_count}_OF_{available_total}", f"CIRCUIT_OPEN_{circuit_open_count}"])
        else:
            # Normal fallback - require majority of checks to pass
            allow = total_checks > 0 and ok_count >= max(1, (total_checks + 1) // 2)
            decision = "GRANT_FULL" if allow else "DENY"
            scope = "FULL_ACCESS" if allow else "NONE"
            risk_score = min(100, (total_checks - ok_count) * 25) if total_checks > 0 else 100
            reason_codes.append(f"CHECKS_{ok_count}_OF_{total_checks}")

        return VerifyResponse(
            decision=decision,
            scope=scope,
            risk_score=risk_score,
            reason_codes=reason_codes,
            allow=allow,
            operator=operator,
            phone_masked=mask_phone(phone),
            latency_ms=(time.time() - start_time) * 1000
        )

# ============================================================================
# WORKLOAD DISCOVERY
# ============================================================================

async def get_active_workloads() -> List[WorkloadInfo]:
    """Get list of active workloads from EAST API."""
    global HTTP_CLIENT
    try:
        # Use shared client if available, otherwise create temporary one
        client = HTTP_CLIENT or httpx.AsyncClient()
        try:
            response = await client.get(f"{EAST_API_URL}/api/workloads", timeout=5.0)
            if response.status_code != 200:
                return []

            data = response.json()
            workloads = []

            # Handle Payload CMS response format
            docs = data.get("docs", data) if isinstance(data, dict) else data

            for w in docs:
                if w.get("status") == "active" and w.get("port"):
                    # Skip self (camara-zero-trust)
                    if "camara" in w.get("usecase", "").lower():
                        continue

                    devices = [d.get("device", "") for d in w.get("devices", [])]
                    workloads.append(WorkloadInfo(
                        id=w.get("id", 0),
                        usecase=w.get("usecase", ""),
                        model=w.get("model", ""),
                        status=w.get("status", ""),
                        port=w.get("port"),
                        devices=devices
                    ))

            return workloads
        finally:
            # Only close if we created a temporary client
            if HTTP_CLIENT is None:
                await client.aclose()
    except Exception as e:
        logging.error(f"Failed to get workloads: {e}")
        return []

async def find_target_port(request: ProxyRequest) -> Optional[int]:
    """Resolve target worker port from request parameters.

    SECURITY: All ports are validated against known EAST workloads
    to prevent SSRF attacks targeting arbitrary localhost services.
    """
    workloads = await get_active_workloads()
    valid_ports = {w.port for w in workloads if w.port}

    # Direct port specified - MUST be validated against known workloads
    if request.target_port:
        if request.target_port not in valid_ports:
            logging.warning(f"[SECURITY] Rejected invalid target_port {request.target_port} - not in known workloads")
            return None
        return request.target_port

    # By workload ID
    if request.target_workload_id:
        for w in workloads:
            # Handle both int and string IDs
            if str(w.id) == str(request.target_workload_id):
                return w.port
        return None

    # By usecase name
    if request.target_usecase:
        usecase_lower = request.target_usecase.lower()
        for w in workloads:
            if usecase_lower in w.usecase.lower():
                return w.port
        return None

    # Default: first text-generation workload
    for w in workloads:
        if "text" in w.usecase.lower() and "generation" in w.usecase.lower():
            return w.port

    # Fallback: any active workload
    if workloads:
        return workloads[0].port

    return None

# ============================================================================
# PROXY LOGIC
# ============================================================================

async def proxy_to_worker(port: int, prompt: str, max_tokens: int, extra_params: Dict = None) -> Dict[str, Any]:
    """Forward inference request to target worker.

    Uses shared HTTP client for connection pooling.
    Timeout reduced to 60s (from 120s) for faster failure detection.
    """
    global HTTP_CLIENT
    url = f"http://127.0.0.1:{port}/infer"

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    if extra_params:
        payload.update(extra_params)

    try:
        # Use shared client if available
        if HTTP_CLIENT:
            response = await HTTP_CLIENT.post(url, json=payload, timeout=60.0)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=60.0)

            if response.status_code != 200:
                return {
                    "error": f"Worker returned {response.status_code}",
                    "detail": response.text[:500]
                }

            return response.json()

        if response.status_code != 200:
            return {
                "error": f"Worker returned {response.status_code}",
                "detail": response.text[:500]
            }

        return response.json()
    except httpx.ConnectError:
        return {"error": f"Cannot connect to worker on port {port}"}
    except httpx.TimeoutException:
        return {"error": "Worker timeout (60s)"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# MODEL SETUP (Optional - for local inference)
# ============================================================================

def setup_model(args) -> bool:
    """Load OpenVINO model for local inference."""
    global PIPE
    
    if not OPENVINO_AVAILABLE:
        logging.info("OpenVINO not available - proxy-only mode")
        return False
    
    if not args.model_name:
        logging.info("No model specified - proxy-only mode")
        return False
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = MODELS_DIR / args.model_name
    
    if not model_path.exists():
        logging.info(f"Model {model_path} not found - proxy-only mode")
        return False
    
    try:
        PIPE = openvino_genai.LLMPipeline(str(model_path), args.device)
        logging.info(f"Model loaded: {args.model_name} on {args.device}")
        return True
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        return False

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CAMARA Zero Trust Proxy for EAST")
    parser.add_argument("--model-name", type=str, default="", help="OpenVINO model (optional)")
    parser.add_argument("--device", type=str, default="CPU", help="Device (CPU, GPU, NPU)")
    parser.add_argument("--port", type=int, default=6005, help="Worker port")
    parser.add_argument("--id", type=int, default=1, help="Workload ID for EAST")
    return parser.parse_args()

def create_app(args):
    global POLICY_ENGINE, AUDIT_PLANE
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global POLICY_ENGINE, AUDIT_PLANE, HTTP_CLIENT

        # Initialize shared HTTP client for connection pooling
        HTTP_CLIENT = httpx.AsyncClient(timeout=30.0)

        # Initialize CAMARA components
        if V6_AVAILABLE:
            POLICY_ENGINE = PolicyEngine(POLICY_FILE)
            AUDIT_PLANE = AuditPlane()
            logging.info("CAMARA policy engine loaded")

        # Load model if specified (optional for proxy mode)
        if args.model_name:
            setup_model(args)

        # Update EAST status
        update_payload_status(args.id, "active", args.port)

        yield

        # Cleanup
        await MCP_MANAGER.disconnect()
        if HTTP_CLIENT:
            await HTTP_CLIENT.aclose()
        update_payload_status(args.id, "stopped", args.port)
    
    app = FastAPI(
        title="CAMARA Zero Trust Proxy",
        description="Zero Trust AI inference gateway for Intel EAST with circuit breaker and rate limiting",
        version="2.1.0",
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==========================================================================
    # ENDPOINTS
    # ==========================================================================
    
    @app.get("/")
    async def root():
        return {
            "worker": "camara-zero-trust-proxy",
            "version": "2.0.0",
            "status": "running",
            "mcp": MCP_MANAGER.connected,
            "model_loaded": PIPE is not None,
            "mode": "proxy" if PIPE is None else "proxy+local"
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "mcp_connected": MCP_MANAGER.connected,
            "model_loaded": PIPE is not None,
            "v6_available": V6_AVAILABLE,
            "proxy_mode": True,
            "circuit_breaker": MCP_CIRCUIT_BREAKER.get_status() if MCP_CIRCUIT_BREAKER else None,
        }

    @app.get("/status")
    async def status():
        """Detailed status including circuit breaker and rate limiter stats."""
        return {
            "worker": "camara-zero-trust-proxy",
            "version": "2.1.0",
            "mcp_connected": MCP_MANAGER.connected,
            "circuit_breaker": MCP_CIRCUIT_BREAKER.get_status() if MCP_CIRCUIT_BREAKER else None,
            "rate_limiters": {
                "verify": VERIFY_RATE_LIMITER.get_stats() if VERIFY_RATE_LIMITER else None,
                "proxy": PROXY_RATE_LIMITER.get_stats() if PROXY_RATE_LIMITER else None,
            },
            "policy_engine": POLICY_ENGINE is not None,
            "audit_plane": AUDIT_PLANE is not None,
        }
    
    @app.get("/workloads", response_model=List[WorkloadInfo])
    async def list_workloads():
        """List available target workloads for proxying."""
        return await get_active_workloads()
    
    @app.post("/verify", response_model=VerifyResponse)
    async def verify(request: VerifyRequest):
        """CAMARA verification only (no inference)."""
        # Rate limiting
        if VERIFY_RATE_LIMITER:
            allowed, retry_after = await VERIFY_RATE_LIMITER.check(request.phone_number)
            if not allowed:
                logging.warning(f"[RATE_LIMIT] Verify rate limit exceeded for {mask_phone(request.phone_number)}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {retry_after:.2f}s",
                    headers={"Retry-After": str(int(retry_after) + 1)}
                )

        if not MCP_MANAGER.connected:
            await MCP_MANAGER.connect()

        return await verify_phone(
            request.phone_number,
            request.operator,
            request.sensitivity
        )
    
    @app.post("/proxy", response_model=ProxyResponse)
    async def proxy(request: ProxyRequest):
        """
        MAIN ENDPOINT: Verify phone, then proxy to target worker.

        This is the Zero Trust gateway - all inference should go through here.
        Includes rate limiting and circuit breaker protection.
        """
        total_start = time.time()

        # Rate limiting (more restrictive for proxy/inference)
        if PROXY_RATE_LIMITER:
            allowed, retry_after = await PROXY_RATE_LIMITER.check(request.phone_number)
            if not allowed:
                logging.warning(f"[RATE_LIMIT] Proxy rate limit exceeded for {mask_phone(request.phone_number)}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {retry_after:.2f}s",
                    headers={"Retry-After": str(int(retry_after) + 1)}
                )

        # Lazy connect MCP
        if not MCP_MANAGER.connected:
            await MCP_MANAGER.connect()

        # Step 1: CAMARA verification
        logging.info(f"[PROXY] Verifying {mask_phone(request.phone_number)}")
        verification = await verify_phone(
            request.phone_number,
            request.operator,
            request.sensitivity
        )
        
        # Step 2: Check if blocked
        if not verification.allow:
            logging.warning(f"[PROXY] BLOCKED - {verification.decision}")
            return ProxyResponse(
                verification=verification,
                blocked=True,
                block_reason=f"CAMARA verification failed: {verification.decision}",
                verification_ms=verification.latency_ms,
                total_ms=(time.time() - total_start) * 1000
            )
        
        # Step 3: Find target worker
        target_port = await find_target_port(request)
        if not target_port:
            logging.error("[PROXY] No target worker found")
            return ProxyResponse(
                verification=verification,
                blocked=True,
                block_reason="No target worker available. Start a workload in EAST first.",
                verification_ms=verification.latency_ms,
                total_ms=(time.time() - total_start) * 1000
            )
        
        logging.info(f"[PROXY] Forwarding to worker on port {target_port}")
        
        # Step 4: Proxy to target worker
        infer_start = time.time()
        target_response = await proxy_to_worker(
            target_port,
            request.prompt,
            request.max_tokens,
            request.extra_params
        )
        inference_ms = (time.time() - infer_start) * 1000
        
        # Step 5: Emit audit receipt (EU AI Act Article 14 - Human Oversight)
        receipt_id = None
        audit_failed = False
        if V6_AVAILABLE and AUDIT_PLANE and "error" not in target_response:
            try:
                # Build complete PolicyDecision with ALL required fields
                policy_decision = PolicyDecision(
                    decision=Decision[verification.decision],
                    scope=Scope[verification.scope],
                    risk_score=verification.risk_score,
                    reason_codes=verification.reason_codes,
                    # Required fields that were missing:
                    decision_summary=f"Proxy verification: {verification.decision} with risk score {verification.risk_score}/100 for {request.sensitivity} sensitivity",
                    sensitivity=request.sensitivity,
                    operator=verification.operator,
                )
                receipt = AUDIT_PLANE.emit_receipt(
                    decision=policy_decision,
                    phone=request.phone_number,
                    input_prompt=request.prompt[:500],
                    timestamp_start=datetime.now(timezone.utc),
                    timestamp_end=datetime.now(timezone.utc),
                    model_path=f"proxy:{target_port}",
                    policy_version=POLICY_ENGINE.version if POLICY_ENGINE else "1.0",
                    kill_switch_state="NORMAL",
                    scope_enforcements=[],
                    openvino_telemetry={"proxy_port": target_port},
                    inference_time_ms=inference_ms
                )
                receipt_id = receipt.receipt_id
                logging.info(f"[PROXY] Receipt: {receipt_id}")
            except Exception as e:
                logging.error(f"[AUDIT] Receipt emission failed: {e}")
                audit_failed = True

        # EU AI Act Article 14: Block request if audit trail cannot be created
        if V6_AVAILABLE and AUDIT_PLANE and audit_failed:
            logging.error("[COMPLIANCE] Blocking response - audit receipt required for EU AI Act compliance")
            return ProxyResponse(
                verification=verification,
                blocked=True,
                block_reason="Audit system error - EU AI Act requires audit trail for all AI decisions",
                target_port=target_port,
                verification_ms=verification.latency_ms,
                inference_ms=inference_ms,
                total_ms=(time.time() - total_start) * 1000
            )
        
        total_ms = (time.time() - total_start) * 1000
        logging.info(f"[PROXY] Complete - verify: {verification.latency_ms:.0f}ms, infer: {inference_ms:.0f}ms, total: {total_ms:.0f}ms")
        
        return ProxyResponse(
            verification=verification,
            blocked="error" in target_response,
            block_reason=target_response.get("error") if "error" in target_response else None,
            target_response=target_response if "error" not in target_response else None,
            target_port=target_port,
            verification_ms=verification.latency_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
            receipt_id=receipt_id
        )
    
    @app.post("/infer")
    async def infer(request: InferRequest):
        """
        Local inference (requires model to be loaded).
        For proxy mode, use /proxy endpoint instead.
        """
        if not MCP_MANAGER.connected:
            await MCP_MANAGER.connect()
        
        # Verify
        verification = await verify_phone(
            request.phone_number,
            request.operator,
            request.sensitivity
        )
        
        if not verification.allow:
            return {
                "text": None,
                "verification": verification.model_dump(),
                "blocked": True,
                "block_reason": f"Verification failed: {verification.decision}"
            }
        
        # Local inference
        if PIPE is None:
            return {
                "text": "[No local model - use /proxy endpoint to forward to another worker]",
                "verification": verification.model_dump(),
                "blocked": False,
                "hint": "POST /proxy with target_usecase='text generation'"
            }
        
        try:
            res = PIPE.generate([request.prompt], max_new_tokens=request.max_tokens)
            return {
                "text": str(res),
                "verification": verification.model_dump(),
                "blocked": False,
                "load_time_s": round(res.perf_metrics.get_load_time() / 1e3, 2),
                "generation_time_s": round(res.perf_metrics.get_generate_duration().mean / 1e3, 2),
                "time_to_token_s": round(res.perf_metrics.get_ttft().mean / 1e3, 2),
                "throughput_s": round(res.perf_metrics.get_throughput().mean, 2),
            }
        except Exception as e:
            return {
                "text": None,
                "verification": verification.model_dump(),
                "blocked": True,
                "block_reason": f"Inference error: {str(e)}"
            }
    
    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    print(f"""
================================================================================
           CAMARA Zero Trust Proxy for Intel EAST v2.1
================================================================================
  Mode:     {'Proxy + Local Model' if args.model_name else 'Proxy Only'}
  Port:     {args.port}
  Device:   {args.device}
  Model:    {args.model_name or '(none - proxy mode)'}
  MCP:      {MCP_URL}
  EAST:     {EAST_API_URL}
  V6 Root:  {CAMARA_V6_ROOT}

  Security Features:
    - Circuit Breaker: 5 failures -> open, 30s timeout, 2 successes -> close
    - Rate Limiting:   /verify 5 req/s, /proxy 2 req/s (per phone)
    - SSRF Protection: Ports validated against known workloads
    - Degraded Mode:   Allows general requests when circuit is open

  Endpoints:
    POST /proxy     - Verify + forward to target worker (MAIN)
    POST /verify    - CAMARA verification only
    POST /infer     - Local inference (if model loaded)
    GET  /workloads - List available targets
    GET  /health    - Health check + circuit breaker status
    GET  /status    - Detailed status (circuit breaker + rate limiter stats)
================================================================================
""")
    
    app = create_app(args)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
