"""Network connectivity check for AI-OS Doctor."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import Optional
from urllib.parse import urlparse

from ..runner import DiagnosticResult, DiagnosticSeverity

logger = logging.getLogger(__name__)

# Endpoints to check
ENDPOINTS = [
    ("HuggingFace Hub", "https://huggingface.co", "/api/models?limit=1"),
    ("HuggingFace Datasets", "https://datasets-server.huggingface.co", "/is-valid?dataset=mnist"),
    ("PyPI", "https://pypi.org", "/simple/"),
]

# Timeout for network requests
REQUEST_TIMEOUT = 10


async def check_network(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check network connectivity to essential services."""
    results = []
    
    # Check basic internet connectivity
    internet_result = await _check_internet_connectivity()
    results.append(internet_result)
    
    if internet_result.severity == DiagnosticSeverity.ERROR:
        # No internet, skip other checks
        return results
    
    # Check proxy settings
    proxy_result = _check_proxy_settings()
    if proxy_result:
        results.append(proxy_result)
    
    # Check each endpoint
    for name, base_url, path in ENDPOINTS:
        result = await _check_endpoint(name, base_url, path)
        results.append(result)
    
    # Check HuggingFace token
    hf_token_result = _check_hf_token()
    if hf_token_result:
        results.append(hf_token_result)
    
    return results


async def _check_internet_connectivity() -> DiagnosticResult:
    """Check basic internet connectivity."""
    # Try to resolve a well-known hostname
    test_hosts = [
        ("8.8.8.8", 53),  # Google DNS
        ("1.1.1.1", 53),  # Cloudflare DNS
    ]
    
    for host, port in test_hosts:
        try:
            # Try TCP connection
            loop = asyncio.get_event_loop()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            await loop.run_in_executor(None, sock.connect, (host, port))
            sock.close()
            
            return DiagnosticResult(
                name="Internet Connectivity",
                severity=DiagnosticSeverity.OK,
                message="Connected",
                details={"test_host": host},
            )
        except Exception:
            continue
    
    # Try DNS resolution as fallback
    try:
        socket.gethostbyname("huggingface.co")
        return DiagnosticResult(
            name="Internet Connectivity",
            severity=DiagnosticSeverity.OK,
            message="DNS resolution working",
        )
    except socket.gaierror:
        pass
    
    return DiagnosticResult(
        name="Internet Connectivity",
        severity=DiagnosticSeverity.ERROR,
        message="No internet connection",
        suggestion="Check your network connection and firewall settings",
    )


def _check_proxy_settings() -> Optional[DiagnosticResult]:
    """Check and report proxy settings."""
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]
    proxy_vars_lower = [v.lower() for v in proxy_vars]
    
    proxies = {}
    for var in proxy_vars + proxy_vars_lower:
        value = os.environ.get(var)
        if value:
            proxies[var] = value
    
    if not proxies:
        return None
    
    return DiagnosticResult(
        name="Proxy Settings",
        severity=DiagnosticSeverity.INFO,
        message=f"{len(proxies)} proxy variable(s) configured",
        details={"proxies": proxies},
    )


async def _check_endpoint(name: str, base_url: str, path: str) -> DiagnosticResult:
    """Check connectivity to a specific endpoint."""
    try:
        import httpx
    except ImportError:
        # Fallback to urllib if httpx not available
        return await _check_endpoint_urllib(name, base_url, path)
    
    url = f"{base_url}{path}"
    start_time = time.perf_counter()
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            latency = time.perf_counter() - start_time
            
            details = {
                "url": url,
                "status_code": response.status_code,
                "latency_ms": round(latency * 1000, 0),
            }
            
            if response.status_code == 200:
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.OK,
                    message=f"Reachable ({latency*1000:.0f}ms)",
                    details=details,
                )
            elif response.status_code == 401:
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Authentication required ({response.status_code})",
                    details=details,
                    suggestion="You may need to set HF_TOKEN for private models",
                )
            elif response.status_code in (502, 503, 504):
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Service temporarily unavailable ({response.status_code})",
                    details=details,
                    suggestion="Service may be experiencing issues. Try again later.",
                )
            else:
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Unexpected status: {response.status_code}",
                    details=details,
                )
                
    except httpx.TimeoutException:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.WARNING,
            message=f"Timeout after {REQUEST_TIMEOUT}s",
            details={"url": url},
            suggestion="Check network connection or try again later",
        )
    except httpx.ConnectError as e:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Connection failed: {e}",
            details={"url": url},
            suggestion="Check firewall settings or network configuration",
        )
    except Exception as e:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Error: {e}",
            details={"url": url},
        )


async def _check_endpoint_urllib(name: str, base_url: str, path: str) -> DiagnosticResult:
    """Fallback endpoint check using urllib."""
    import urllib.request
    import urllib.error
    
    url = f"{base_url}{path}"
    start_time = time.perf_counter()
    
    try:
        loop = asyncio.get_event_loop()
        request = urllib.request.Request(url, headers={"User-Agent": "AI-OS-Doctor/1.0"})
        
        def do_request():
            return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)
        
        response = await loop.run_in_executor(None, do_request)
        latency = time.perf_counter() - start_time
        
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.OK,
            message=f"Reachable ({latency*1000:.0f}ms)",
            details={"url": url, "status_code": response.status, "latency_ms": round(latency * 1000, 0)},
        )
        
    except urllib.error.HTTPError as e:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.WARNING,
            message=f"HTTP error: {e.code}",
            details={"url": url, "status_code": e.code},
        )
    except urllib.error.URLError as e:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Connection failed: {e.reason}",
            details={"url": url},
        )
    except Exception as e:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Error: {e}",
            details={"url": url},
        )


def _check_hf_token() -> Optional[DiagnosticResult]:
    """Check HuggingFace token configuration."""
    # Check environment variables
    token_vars = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    token_found = None
    token_var = None
    
    for var in token_vars:
        value = os.environ.get(var)
        if value:
            token_found = value
            token_var = var
            break
    
    # Check huggingface-cli login
    hf_token_file = None
    try:
        from pathlib import Path
        hf_dir = Path.home() / ".huggingface"
        token_file = hf_dir / "token"
        if token_file.exists():
            hf_token_file = str(token_file)
            if not token_found:
                token_found = "file"
    except Exception:
        pass
    
    if token_found:
        # Mask the token for display
        if token_found != "file":
            masked = token_found[:4] + "..." + token_found[-4:] if len(token_found) > 8 else "***"
        else:
            masked = "(from file)"
        
        details = {"source": token_var or "file"}
        if hf_token_file:
            details["token_file"] = hf_token_file
        
        return DiagnosticResult(
            name="HuggingFace Token",
            severity=DiagnosticSeverity.OK,
            message=f"Configured {masked}",
            details=details,
        )
    else:
        return DiagnosticResult(
            name="HuggingFace Token",
            severity=DiagnosticSeverity.INFO,
            message="Not configured",
            suggestion="Set HF_TOKEN for private models: huggingface-cli login",
        )
