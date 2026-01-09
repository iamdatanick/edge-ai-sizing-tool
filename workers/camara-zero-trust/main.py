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

import re

# Phone validation pattern
PHONE_PATTERN = re.compile(r'^\+[1-9]\d{6,14}$')

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
    """Manage MCP connection lifecycle."""
    def __init__(self):
        self.session = None
        self.connected = False
        self._sse_context = None
        self._session_context = None
    
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
            logging.info("MCP connected")
            return True
        except Exception as e:
            logging.error(f"MCP connection failed: {e}")
            self.connected = False
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
    
    async def call_tool(self, tool: str, params: dict) -> Tuple[str, str, float]:
        if not self.connected or not self.session:
            return ("ERROR", "MCP not connected", 0)
        
        start = time.time()
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool, params),
                timeout=15.0
            )
            text = result.content[0].text if result.content else ""
            elapsed = (time.time() - start) * 1000
            return ("OK", text[:200], elapsed)
        except asyncio.TimeoutError:
            return ("TIMEOUT", "Timeout", (time.time() - start) * 1000)
        except Exception as e:
            return ("ERROR", str(e)[:100], (time.time() - start) * 1000)

MCP_MANAGER = MCPManager()

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
        # Fallback evaluation - require majority of checks to pass
        ok_count = sum(1 for e in evidence if
                      (e.status if hasattr(e, 'status') else e.get('status')) == "OK")
        total_checks = len(evidence)
        # Require majority to pass, minimum 1 (handles single-check operators)
        allow = total_checks > 0 and ok_count >= max(1, (total_checks + 1) // 2)
        return VerifyResponse(
            decision="GRANT_FULL" if allow else "DENY",
            scope="FULL_ACCESS" if allow else "NONE",
            risk_score=min(100, (total_checks - ok_count) * 25) if total_checks > 0 else 100,
            reason_codes=["FALLBACK_EVAL", f"CHECKS_{ok_count}_OF_{total_checks}"],
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
        description="Zero Trust AI inference gateway for Intel EAST",
        version="2.0.0",
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
            "proxy_mode": True
        }
    
    @app.get("/workloads", response_model=List[WorkloadInfo])
    async def list_workloads():
        """List available target workloads for proxying."""
        return await get_active_workloads()
    
    @app.post("/verify", response_model=VerifyResponse)
    async def verify(request: VerifyRequest):
        """CAMARA verification only (no inference)."""
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
        """
        total_start = time.time()
        
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
           CAMARA Zero Trust Proxy for Intel EAST v2.0
================================================================================
  Mode:     {'Proxy + Local Model' if args.model_name else 'Proxy Only'}
  Port:     {args.port}
  Device:   {args.device}
  Model:    {args.model_name or '(none - proxy mode)'}
  MCP:      {MCP_URL}
  EAST:     {EAST_API_URL}
  V6 Root:  {CAMARA_V6_ROOT}
  
  Endpoints:
    POST /proxy     - Verify + forward to target worker (MAIN)
    POST /verify    - CAMARA verification only
    POST /infer     - Local inference (if model loaded)
    GET  /workloads - List available targets
    GET  /health    - Health check
================================================================================
""")
    
    app = create_app(args)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
