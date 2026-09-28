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
