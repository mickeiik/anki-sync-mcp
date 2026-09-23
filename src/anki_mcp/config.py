from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def validate_sync_endpoint(value: str) -> str:
    """Validate a custom sync endpoint without exposing embedded credentials."""
    if any(character.isspace() for character in value):
        raise ValueError("ANKI_SYNC_HOST must not contain whitespace")
    endpoint = value.strip()
    if not endpoint:
        return ""
    parsed = urlsplit(endpoint)
    if len(endpoint.encode("utf-8")) > 2048:
        raise ValueError("ANKI_SYNC_HOST must not exceed 2048 UTF-8 bytes")
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("ANKI_SYNC_HOST contains an invalid hostname") from exc
    if not parsed.netloc or not hostname or parsed.scheme not in {"http", "https"}:
        raise ValueError("ANKI_SYNC_HOST must be empty or an HTTP(S) base URL")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("ANKI_SYNC_HOST contains an invalid port") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("ANKI_SYNC_HOST must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("ANKI_SYNC_HOST must not contain a query or fragment")
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and parsed.hostname not in loopback_hosts:
        raise ValueError("ANKI_SYNC_HOST requires HTTPS except for loopback development")
    return endpoint


def validate_sync_migration_endpoint(value: str, configured_endpoint: str | None) -> str:
    """Restrict server-directed migrations to an explicitly trusted origin."""
    endpoint = validate_sync_endpoint(value)
    parsed = urlsplit(endpoint)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if configured_endpoint:
        configured = urlsplit(configured_endpoint)
        configured_hostname = (configured.hostname or "").casefold().rstrip(".")
        configured_port = configured.port or (443 if configured.scheme == "https" else 80)
        if (parsed.scheme, hostname, port) != (
            configured.scheme,
            configured_hostname,
            configured_port,
        ):
            raise ValueError("sync endpoint migration left the configured trusted origin")
    elif parsed.scheme != "https" or not (
        hostname == "ankiweb.net" or hostname.endswith(".ankiweb.net")
    ):
        raise ValueError("AnkiWeb endpoint migration targeted an untrusted origin")
    return endpoint


class Settings(BaseSettings):
    """Validated process configuration.

    Token material is resolved once at startup and retained as a SecretStr.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=True,
        extra="forbid",
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    host: str = Field("0.0.0.0", alias="MCP_HOST")
    port: int = Field(8000, ge=1, le=65535, alias="MCP_PORT")
    mcp_path: str = Field("/mcp", alias="MCP_PATH")
    auth_token_input: SecretStr | None = Field(None, alias="MCP_AUTH_TOKEN")
    auth_token_file: Path | None = Field(None, alias="MCP_AUTH_TOKEN_FILE")
    collection_path: Path = Field(Path("/data/collection.anki2"), alias="ANKI_COLLECTION_PATH")
    sync_username: str = Field("", alias="ANKI_SYNC_USERNAME")
    sync_password_input: SecretStr | None = Field(None, alias="ANKI_SYNC_PASSWORD")
    sync_password_file: Path | None = Field(None, alias="ANKI_SYNC_PASSWORD_FILE")
    sync_host: str = Field("", alias="ANKI_SYNC_HOST")
    scopes_csv: str = Field("read,write,admin", alias="MCP_SCOPES")
    sync_on_read: bool = Field(False, alias="ANKI_SYNC_ON_READ")
    sync_on_write: bool = Field(True, alias="ANKI_SYNC_ON_WRITE")
    allow_destructive: bool = Field(False, alias="ANKI_ALLOW_DESTRUCTIVE")
    allow_full_sync: bool = Field(False, alias="ANKI_ALLOW_FULL_SYNC")
    allow_restore: bool = Field(False, alias="ANKI_ALLOW_RESTORE")
    allow_schema_changes: bool = Field(False, alias="ANKI_ALLOW_SCHEMA_CHANGES")
    allow_review_answers: bool = Field(False, alias="ANKI_ALLOW_REVIEW_ANSWERS")
    allow_undo: bool = Field(False, alias="ANKI_ALLOW_UNDO")
    confirmation_ttl_seconds: int = Field(
        300, ge=30, le=3600, alias="ANKI_CONFIRMATION_TTL_SECONDS"
    )
    bootstrap_mode: Literal["disabled", "download_if_empty"] = Field(
        "disabled", alias="ANKI_BOOTSTRAP_MODE"
    )
    max_batch_size: int = Field(50, ge=1, le=500, alias="ANKI_MAX_BATCH_SIZE")
    max_page_size: int = Field(100, ge=1, le=1000, alias="MCP_MAX_PAGE_SIZE")
    max_search_scan: int = Field(10_000, ge=1, le=1_000_000, alias="MCP_MAX_SEARCH_SCAN")
    max_rendered_field_bytes: int = Field(
        262_144, ge=1, le=4_194_304, alias="MCP_MAX_RENDERED_FIELD_BYTES"
    )
    max_card_fields: int = Field(100, ge=1, le=1000, alias="MCP_MAX_CARD_FIELDS")
    max_media_bytes: int = Field(1_048_576, ge=1, le=16_777_216, alias="ANKI_MAX_MEDIA_BYTES")
    sync_timeout_seconds: float = Field(300, gt=0, le=3600, alias="ANKI_SYNC_TIMEOUT_SECONDS")
    max_response_bytes: int = Field(
        1_048_576, ge=1024, le=16_777_216, alias="MCP_MAX_RESPONSE_BYTES"
    )
    max_request_bytes: int = Field(1_048_576, ge=1024, le=16_777_216, alias="MCP_MAX_REQUEST_BYTES")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    allowed_hosts_csv: str = Field(
        "testserver,localhost,localhost:*,127.0.0.1,127.0.0.1:*,anki-mcp,anki-mcp:*",
        alias="MCP_ALLOWED_HOSTS",
    )
    allowed_origins_csv: str = Field(
        "http://localhost:*,http://127.0.0.1:*", alias="MCP_ALLOWED_ORIGINS"
    )
    auth_token: SecretStr = Field(default=SecretStr(""), exclude=True)
    sync_password: SecretStr | None = Field(default=None, exclude=True)

    @property
    def sync_endpoint(self) -> str | None:
        endpoint = self.sync_host.strip()
        return endpoint or None

    @field_validator("sync_host")
    @classmethod
    def valid_sync_host(cls, value: str) -> str:
        return validate_sync_endpoint(value)

    @field_validator("mcp_path")
    @classmethod
    def valid_mcp_path(cls, value: str) -> str:
        if not value.startswith("/") or value == "/":
            raise ValueError("MCP_PATH must be an absolute, non-root path")
        if any(character in value for character in "{}?#") or any(
            character.isspace() for character in value
        ):
            raise ValueError("MCP_PATH must be a static URL path without templates or whitespace")
        value = value.rstrip("/")
        if value == "/health" or value.startswith("/health/"):
            raise ValueError("MCP_PATH must not overlap the reserved /health routes")
        return value

    @property
    def allowed_hosts(self) -> list[str]:
        return [item.strip() for item in self.allowed_hosts_csv.split(",") if item.strip()]

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins_csv.split(",") if item.strip()]

    @property
    def scopes(self) -> frozenset[str]:
        return frozenset(item.strip() for item in self.scopes_csv.split(",") if item.strip())

    @field_validator("scopes_csv")
    @classmethod
    def valid_scopes(cls, value: str) -> str:
        scopes = {item.strip() for item in value.split(",") if item.strip()}
        unknown = scopes - {"read", "write", "admin", "destructive"}
        if unknown:
            raise ValueError(f"unknown MCP scope: {', '.join(sorted(unknown))}")
        return value

    @model_validator(mode="after")
    def validate_response_budget(self) -> Self:
        minimum = self.max_rendered_field_bytes + 4096
        if self.max_response_bytes < minimum:
            raise ValueError(
                "MCP_MAX_RESPONSE_BYTES must be at least MCP_MAX_RENDERED_FIELD_BYTES + 4096"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def resolve_secrets(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)

        def resolve(
            direct_alias: str,
            direct_field: str,
            file_alias: str,
            file_field: str,
            description: str,
            required: bool = True,
        ) -> SecretStr | None:
            direct = values.get(direct_alias, values.get(direct_field))
            file_value = values.get(file_alias, values.get(file_field))
            if direct is None and file_value is None and not required:
                return None
            if (direct is not None) == (file_value is not None):
                raise ValueError(f"exactly one of {direct_alias} and {file_alias} is required")
            if direct is not None:
                raw = direct.get_secret_value() if isinstance(direct, SecretStr) else str(direct)
            else:
                try:
                    raw = Path(str(file_value)).read_text(encoding="utf-8").removesuffix("\n")
                except OSError as exc:
                    raise ValueError(f"unable to read {file_alias}") from exc
            if not raw:
                raise ValueError(f"{description} must not be empty")
            return SecretStr(raw)

        values["auth_token"] = resolve(
            "MCP_AUTH_TOKEN",
            "auth_token_input",
            "MCP_AUTH_TOKEN_FILE",
            "auth_token_file",
            "MCP authentication token",
        )
        values["sync_password"] = resolve(
            "ANKI_SYNC_PASSWORD",
            "sync_password_input",
            "ANKI_SYNC_PASSWORD_FILE",
            "sync_password_file",
            "Anki sync password",
            required=False,
        )
        return values
