import re
from typing import Optional, Tuple
from app.api.schemas import ChatCompletionRequest


# High-risk PII and sensitive token regular expressions
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Standard international and domestic phone numbers
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
)

# US Social Security Number (SSN): AAA-GG-SSSS
SSN_REGEX = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

# Credit card numbers (Visa, MasterCard, Amex, Discover: 13 to 19 digits with optional hyphens/spaces)
CREDIT_CARD_REGEX = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b|"
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)

# Common API Keys and secret tokens (OpenAI, GitHub, AWS, generic high-entropy bearer tokens)
API_KEY_REGEX = re.compile(
    r"\b(?:sk-[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16}|Bearer\s+[a-zA-Z0-9._-]{24,})\b",
    re.IGNORECASE,
)


def contains_pii(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scans text for high-risk PII and sensitive credentials.
    Returns: (is_detected, pii_type_label)
    """
    if not text:
        return False, None

    if EMAIL_REGEX.search(text):
        return True, "EMAIL"

    if SSN_REGEX.search(text):
        return True, "SSN"

    if API_KEY_REGEX.search(text):
        return True, "API_KEY"

    # Credit card check: verify digit length between 13 and 19
    cc_match = CREDIT_CARD_REGEX.search(text)
    if cc_match:
        digits_only = re.sub(r"\D", "", cc_match.group(0))
        if 13 <= len(digits_only) <= 19:
            return True, "CREDIT_CARD"

    # Phone number check: ensure minimum 10 digits to prevent matching small numbers
    phone_matches = PHONE_REGEX.findall(text)
    for match in phone_matches:
        digits_only = re.sub(r"\D", "", match)
        if len(digits_only) >= 10:
            return True, "PHONE"

    return False, None


def check_request_pii(request: ChatCompletionRequest) -> Tuple[bool, Optional[str]]:
    """
    Checks all messages in a ChatCompletionRequest for sensitive PII.
    """
    for msg in request.messages:
        has_pii, pii_type = contains_pii(msg.content)
        if has_pii:
            return True, pii_type
    return False, None
