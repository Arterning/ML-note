"""安全解析 EML，并提取文本和轻量结构化特征。"""

from __future__ import annotations

import hashlib
import html
import re
from email import policy
from email.message import EmailMessage, Message
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr
from html.parser import HTMLParser
from pathlib import Path


URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SUSPICIOUS_EXTENSIONS = {
    ".bat", ".cmd", ".com", ".exe", ".hta", ".iso", ".js",
    ".jse", ".lnk", ".msi", ".ps1", ".scr", ".vbs", ".wsf",
}

HAM_DIRECTORY_NAMES = {"ham", "normal", "正常", "正常邮件"}
SPAM_DIRECTORY_NAMES = {"spam", "junk", "垃圾", "垃圾邮件"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data):
        if not self.ignored_depth:
            self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(value)
        text = " ".join(parser.parts)
    except Exception:
        text = HTML_TAG_PATTERN.sub(" ", value)
    return normalize_whitespace(html.unescape(text))


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def _safe_content(part: Message) -> str:
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except (LookupError, UnicodeDecodeError, AttributeError):
        pass

    payload = part.get_payload(decode=True)
    if not payload:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def extract_bodies(message: EmailMessage, max_body_chars: int) -> tuple[str, str]:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type().lower()
        if content_type == "text/plain":
            plain_parts.append(_safe_content(part))
        elif content_type == "text/html":
            html_parts.append(_safe_content(part))

    plain_body = normalize_whitespace(" ".join(plain_parts))
    raw_html = " ".join(html_parts)
    html_body = html_to_text(raw_html)
    body = plain_body or html_body
    return body[:max_body_chars], raw_html[:max_body_chars]


def _domain(address_header: str) -> str:
    address = parseaddr(address_header)[1].lower()
    return address.rsplit("@", 1)[-1] if "@" in address else ""


def infer_label(path: Path, input_root: Path) -> str:
    try:
        parts = path.relative_to(input_root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    normalized = {part.strip().lower() for part in parts}
    if normalized & SPAM_DIRECTORY_NAMES:
        return "spam"
    if normalized & HAM_DIRECTORY_NAMES:
        return "ham"
    return ""


def template_fingerprint(text: str) -> str:
    normalized = text.lower()
    normalized = URL_PATTERN.sub(" <url> ", normalized)
    normalized = EMAIL_PATTERN.sub(" <email> ", normalized)
    normalized = NUMBER_PATTERN.sub(" <number> ", normalized)
    normalized = normalize_whitespace(normalized)
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def parse_eml(path: Path, input_root: Path | None = None, max_body_chars: int = 50_000) -> dict:
    """解析单个 EML；不执行附件，也不提取附件正文。"""
    with path.open("rb") as file_handle:
        message = BytesParser(policy=policy.default).parse(file_handle)

    subject = normalize_whitespace(str(message.get("Subject", "")))
    body, raw_html = extract_bodies(message, max_body_chars=max_body_chars)
    text = normalize_whitespace(f"{subject}\n{body}")
    attachment_names = [
        part.get_filename() or ""
        for part in message.iter_attachments()
    ]
    suspicious_attachment_count = sum(
        Path(filename).suffix.lower() in SUSPICIOUS_EXTENSIONS
        for filename in attachment_names
    )
    sender_domain = _domain(str(message.get("From", "")))
    reply_domain = _domain(str(message.get("Reply-To", "")))
    recipients = getaddresses(message.get_all("To", []) + message.get_all("Cc", []))
    alphabetic = [character for character in text if character.isalpha()]
    uppercase_count = sum(character.isupper() for character in alphabetic)
    content_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    root = input_root or path.parent
    try:
        email_id = path.relative_to(root).as_posix()
    except ValueError:
        email_id = str(path.resolve())

    return {
        "email_id": email_id,
        "source_path": str(path.resolve()),
        "label": infer_label(path, root),
        "subject": subject,
        "body": body,
        "text": text,
        "sender_domain": sender_domain,
        "content_hash": content_hash,
        "template_group": template_fingerprint(text),
        "text_length": len(text),
        "subject_length": len(subject),
        "url_count": len(URL_PATTERN.findall(text)),
        "recipient_count": len(recipients),
        "attachment_count": len(attachment_names),
        "suspicious_attachment_count": suspicious_attachment_count,
        "exclamation_count": text.count("!") + text.count("！"),
        "uppercase_ratio": uppercase_count / max(1, len(alphabetic)),
        "has_html": int(bool(raw_html)),
        "reply_to_mismatch": int(bool(reply_domain and sender_domain != reply_domain)),
    }
