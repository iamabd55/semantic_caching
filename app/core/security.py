from fastapi import Request


def get_tenant_id(request: Request) -> str:
    """
    Resolve the tenant identity for the current request.

    SECURITY ARCHITECTURE NOTE:
    In a full production deployment, tenant identity MUST NOT be trusted directly from
    arbitrary user headers. It must be resolved server-side from a cryptographically
    verified JWT claim (e.g., token['tenant_id']), an authenticated API key mapping,
    or a secure session cookie after SSO/OAuth authentication.

    For this gateway infrastructure layer and portfolio test harness, the tenant is
    resolved from the 'X-Tenant-ID' request header (falling back to 'default').
    """
    header_val = request.headers.get("X-Tenant-ID")
    if header_val and header_val.strip():
        # Sanitize to alphanumeric, dashes, and underscores
        clean_tenant = "".join(
            c for c in header_val.strip().lower() if c.isalnum() or c in ("-", "_")
        )
        return clean_tenant or "default"

    return "default"


def get_cache_ttl(request: Request) -> int:
    """
    Resolve the Time-To-Live (TTL in seconds) for the current request.
    Extracts custom TTL from 'X-Cache-TTL' or 'Cache-Control: max-age=...' headers;
    otherwise falls back to settings.DEFAULT_CACHE_TTL_SECONDS.
    """
    from app.core.config import settings

    ttl_header = request.headers.get("X-Cache-TTL")
    if ttl_header and ttl_header.strip():
        try:
            val = int(ttl_header.strip())
            if val >= 0:
                return val
        except ValueError:
            pass

    cache_control = request.headers.get("Cache-Control")
    if cache_control and "max-age=" in cache_control:
        for part in cache_control.split(","):
            part = part.strip()
            if part.startswith("max-age="):
                try:
                    val = int(part.split("=")[1])
                    if val >= 0:
                        return val
                except ValueError:
                    pass

    return settings.DEFAULT_CACHE_TTL_SECONDS
