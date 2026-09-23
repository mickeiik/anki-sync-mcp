# anki-sync-mcp

**A headless, sync-native Anki MCP sidecar for always-on AI agents.**

`anki-sync-mcp` lets an MCP client safely manage a real Anki collection without Anki
Desktop or AnkiConnect. It runs as an authenticated Streamable HTTP service, uses Anki's
collection and sync APIs, and synchronizes with AnkiWeb or a self-hosted Anki sync server.

Unlike desktop bridges, it is designed to run unattended in Docker. Writes are serialized,
idempotent, persisted across restarts, and synchronized before and after mutation. Destructive
and schema-changing tools are disabled by default and use preview tokens plus verified backups
when enabled.

It supports decks, notes, cards, tags, note types, templates, media, scheduling controls,
review history and analytics, backups, and explicit full-sync recovery. Tag tools cover listing
(optionally with note counts that include child tags), renaming, deleting, and merging a tag and
its children into another tag.

## Quick start with Docker Compose

Requirements: Docker Compose, Python 3 for token generation, and an AnkiWeb or self-hosted
sync account.

```sh
git clone https://github.com/raslab/anki-sync-mcp.git
cd anki-sync-mcp

mkdir -p secrets
python -c 'import secrets; print(secrets.token_urlsafe(32))' > secrets/anki_mcp_token
read -rsp "Anki sync password: " ANKI_PASSWORD
printf '%s' "$ANKI_PASSWORD" > secrets/anki_sync_password
unset ANKI_PASSWORD
chmod 600 secrets/*

export ANKI_SYNC_USERNAME="your-sync-username"
export ANKI_SYNC_HOST="" # empty for AnkiWeb; otherwise an HTTPS sync-server URL

docker compose up -d --build
docker compose exec anki-mcp anki-mcp-healthcheck
```

The included Compose configuration exposes MCP only to containers on its private `assistant`
network. Connect your agent container to that network and configure:

- URL: `http://anki-mcp:8000/mcp`
- Header: `Authorization: Bearer <contents of secrets/anki_mcp_token>`

For a client running directly on the host, add this to the `anki-mcp` service in
`compose.yaml`, then run `docker compose up -d` again:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

Use `http://127.0.0.1:8000/mcp` as the MCP URL. After connecting, call
`anki_sync_login` once. The resulting sync host key is stored in the persistent Docker volume,
so you can later remove `ANKI_SYNC_PASSWORD_FILE` from `compose.yaml` and delete the password
secret when automatic reauthentication is not required.

Do not run two sidecars, Anki Desktop, or any other process against the same collection file.
Keep destructive, schema, and full-sync feature flags disabled except during deliberate
maintenance.

## Configuration

Configuration is read from environment variables when the service starts. `compose.yaml`
supplies safe defaults and mounts the authentication token and initial sync password as Docker
secrets. Variables written as `${NAME:-default}` can be overridden through the shell or a
Compose `.env` file. To change a fixed value, or use one of the settings not present in the
Compose file, add it under `services.anki-mcp.environment`.

### Server and access

| Variable | Default | Description |
| --- | --- | --- |
| `MCP_HOST` | `0.0.0.0` | Address on which the HTTP server listens. |
| `MCP_PORT` | `8000` | HTTP listen port (`1`-`65535`). Keep this consistent with `expose`, any `ports` mapping, and the client URL. |
| `MCP_PATH` | `/mcp` | MCP endpoint path. It must be an absolute, static, non-root path and must not overlap `/health`. |
| `MCP_AUTH_TOKEN` | required alternative | Bearer token supplied directly. Use exactly one of this setting and `MCP_AUTH_TOKEN_FILE`; the Compose setup uses the file variant. |
| `MCP_AUTH_TOKEN_FILE` | required alternative | Path to a file containing the bearer token. One trailing newline is ignored. Compose mounts `./secrets/anki_mcp_token` at `/run/secrets/anki_mcp_token`. |
| `MCP_SCOPES` | `read,write,admin` | Comma-separated tool groups to register. Allowed values are `read`, `write`, `admin`, and `destructive`. Destructive tools require both the `destructive` scope and their feature flag. |
| `MCP_ALLOWED_HOSTS` | `testserver,localhost,localhost:*,127.0.0.1,127.0.0.1:*,anki-mcp,anki-mcp:*` | Comma-separated HTTP `Host` patterns accepted by MCP DNS-rebinding protection. Add the service name or hostname used by remote clients. |
| `MCP_ALLOWED_ORIGINS` | `http://localhost:*,http://127.0.0.1:*` | Comma-separated browser origins accepted by MCP transport security. Add explicit trusted origins when using a browser-based client. |
| `LOG_LEVEL` | `INFO` | Uvicorn log level, such as `debug`, `info`, `warning`, or `error`. |

The `/health/live` and `/health/ready` endpoints are not bearer-authenticated. The supplied
Compose network keeps them private unless you publish the container port.

### Collection and synchronization

| Variable | Default | Description |
| --- | --- | --- |
| `ANKI_COLLECTION_PATH` | `/data/collection.anki2` | Local Anki collection file. Compose persists its parent directory in the `anki_data` volume; do not point multiple processes at the same file. |
| `ANKI_SYNC_USERNAME` | empty | AnkiWeb username or self-hosted sync username. Required for `anki_sync_login` and for bootstrap when no persisted authentication exists. |
| `ANKI_SYNC_PASSWORD` | optional alternative | Sync password supplied directly. It is optional after authentication has been persisted; do not set it together with `ANKI_SYNC_PASSWORD_FILE`. |
| `ANKI_SYNC_PASSWORD_FILE` | optional alternative | Path to the sync-password file. Compose mounts `./secrets/anki_sync_password` at `/run/secrets/anki_sync_password`. Remove the setting and secret after login if automatic reauthentication is not needed. |
| `ANKI_SYNC_HOST` | empty (AnkiWeb) | Custom sync-server HTTP(S) base URL. Non-loopback servers require HTTPS; credentials, query strings, and fragments are rejected. |
| `ANKI_SYNC_ON_READ` | `false` | Synchronize before coordinated reads. Individual read tools can still request a sync with `sync_before=true`. |
| `ANKI_SYNC_ON_WRITE` | `true` | Synchronize before and after coordinated mutations. Disabling this permits local changes to remain unsynchronized. |
| `ANKI_SYNC_TIMEOUT_SECONDS` | `300` | Maximum wait for media synchronization (`> 0`, up to `3600` seconds). |
| `ANKI_BOOTSTRAP_MODE` | `disabled` | Startup behavior: `disabled`, or `download_if_empty` to authenticate and perform a download-only initialization when the local collection is empty. Startup fails rather than uploading if the server requests a full upload. |

The persisted `anki_data` volume also contains synchronization authentication state, mutation
receipts, and backups. Preserve the volume across container recreation.

### Safety and resource limits

| Variable | Default | Description |
| --- | --- | --- |
| `ANKI_ALLOW_DESTRUCTIVE` | `false` | Registers delete, tag-merge, and FSRS rescheduling preview/apply tools when `MCP_SCOPES` also includes `destructive`. Applying destructive changes requires a preview token and verified fresh backup. |
| `ANKI_ALLOW_FULL_SYNC` | `false` | Registers explicit full-download and full-upload recovery tools. It must also be enabled for schema-changing tools. |
| `ANKI_ALLOW_RESTORE` | `false` | Registers backup listing and restore tools when `MCP_SCOPES` includes `destructive`; restore additionally requires `ANKI_ALLOW_FULL_SYNC`. Applying a restore requires a preview token and creates a verified pre-restore backup. The restored client must then either overwrite the server (`anki_sync_full_upload(force=true)`) or be reconciled deliberately, because a normal sync would merge the server's pre-restore state onto the restored backup. |
| `ANKI_ALLOW_SCHEMA_CHANGES` | `false` | Registers note-type and template mutation tools when `ANKI_ALLOW_FULL_SYNC` is also enabled. Note-type deletion additionally requires destructive access. |
| `ANKI_ALLOW_REVIEW_ANSWERS` | `false` | Registers the administrative card-answering tool, which changes scheduling state. |
| `ANKI_ALLOW_UNDO` | `false` | Registers the `anki_undo` and `anki_redo` tools when `MCP_SCOPES` also includes `destructive` and `ANKI_SYNC_ON_WRITE=false`. Undo history is in-memory, is cleared by a successful synchronization, and is not durable across restarts, so with the default `ANKI_SYNC_ON_WRITE=true` (where every mutation syncs) the two tools are not registered at all. `anki_undo_status` stays available and reports a `disabled_reason` naming every blocking setting. |
| `ANKI_ALLOW_IMPORT` | `false` | Registers the destructive-scope `.apkg`/`.csv` import tools when `MCP_SCOPES` also includes `destructive`. Files must be placed in the collection's `imports/` folder (for the container: `/data/imports/`, e.g. with `docker cp`), are size-capped by `ANKI_MAX_IMPORT_BYTES`, and require a preview token plus a verified pre-operation backup. `.apkg` import additionally requires `ANKI_ALLOW_SCHEMA_CHANGES` and `ANKI_ALLOW_FULL_SYNC`, because packages can add note types and change scheduling. The CSV preview token binds the file identity (content digest) and delimiter, while the note type, field mapping, deck, tags, and duplicate resolution are chosen at apply time by the same authenticated caller. |
| `ANKI_CONFIRMATION_TTL_SECONDS` | `300` | Lifetime of destructive-operation preview tokens (`30`-`3600` seconds). Tokens are single-use. |
| `ANKI_MAX_BATCH_SIZE` | `50` | Maximum notes accepted by one atomic batch-create operation (`1`-`500`). |
| `MCP_MAX_PAGE_SIZE` | `100` | Maximum `limit` accepted by paginated tools (`1`-`1000`). |
| `MCP_MAX_SEARCH_SCAN` | `10000` | Maximum collection items that bounded list/search and tag operations may scan (`1`-`1000000`). |
| `MCP_MAX_RENDERED_FIELD_BYTES` | `262144` | UTF-8 byte limit for each rendered text field in responses (`1`-`4194304`); oversized values are truncated and marked. |
| `MCP_MAX_CARD_FIELDS` | `100` | Maximum fields or templates returned for a note or note type (`1`-`1000`); omitted counts are reported. |
| `ANKI_MAX_MEDIA_BYTES` | `1048576` | Maximum decoded size of a media file read, written, or renamed (`1`-`16777216` bytes). |
| `ANKI_MAX_IMPORT_BYTES` | `268435456` | Caps both the staged file size and, for an `.apkg`, the declared uncompressed size of its members (`1048576`-`4294967296` bytes); files over the cap are rejected before parsing or inflation. |
| `MCP_MAX_RESPONSE_BYTES` | `1048576` | Maximum serialized tool-result size (`1024`-`16777216` bytes). It must be at least `MCP_MAX_RENDERED_FIELD_BYTES + 4096`. |
| `MCP_MAX_REQUEST_BYTES` | `1048576` | Maximum MCP request-body size (`1024`-`16777216` bytes); larger requests receive HTTP 413. |

Boolean values accept the forms supported by Pydantic settings, including `true`/`false`,
`1`/`0`, and `yes`/`no`. Invalid or unknown configuration causes startup to fail instead of
silently falling back to a default.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen --all-groups
uv run pytest
uv run ruff check .
uv run pyright
```

## License

[MIT](LICENSE) — free to use, modify, and distribute under the license terms.

## Trademark disclaimer

This is an independent, unofficial project. It is not affiliated with, endorsed by, or
sponsored by Anki, AnkiWeb, Ankitects Pty Ltd, or their maintainers. Anki and AnkiWeb are
trademarks of their respective owners.