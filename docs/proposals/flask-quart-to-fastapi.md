# RFC: Flask/Quart → FastAPI Migration Feasibility Assessment

- **Status**: Deferred
- **Issue**: #14752
- **Assessment date**: 2026-05-10
- **Decision date**: 2026-05-11

---

## 1. Current State Overview

### 1.1 Framework Distribution

| Area | Framework | Scale |
|------|-----------|-------|
| Main API (`api/`) | Quart (async Flask) | 26 files, ~10k lines |
| Admin Panel (`admin/server/`) | Flask | 7 files, ~2.2k lines |
| **Total** | — | 33 files, ~12.2k lines |

### 1.2 File Distribution

```
api/apps/                          # Main API (Quart)
├── __init__.py                   # 367 lines - app factory + blueprint loading
├── llm_app.py
├── backward_compat.py
├── dify_retrieval.py
├── doc.py (sdk)
├── session.py (sdk)
└── restful_apis/                 # Core REST API
    ├── agent_api.py             # ~1,892 lines
    ├── document_api.py          # ~1,895 lines
    ├── chat_api.py              # ~1,130 lines
    ├── dataset_api.py           # ~810 lines
    ├── user_api.py              # ~842 lines
    ├── file_api.py              # ~376 lines
    ├── connector_api.py         # ~531 lines
    ├── chunk_api.py             # ~445 lines
    ├── search_api.py            # ~216 lines
    ├── system_api.py            # ~421 lines
    ├── mcp_api.py               # ~331 lines
    ├── memory_api.py            # ~304 lines
    ├── openai_api.py            # ~300 lines
    ├── task_api.py              # ~101 lines
    ├── plugin_api.py            # ~30 lines
    ├── stats_api.py             # ~55 lines
    ├── file2document_api.py     # ~176 lines
    ├── langfuse_api.py          # ~99 lines
    └── tenant_api.py            # ~163 lines

admin/server/                     # Admin Panel (Flask)
├── admin_server.py
├── routes.py
├── auth.py
├── services.py
├── config.py
├── roles.py
├── responses.py
├── models.py
└── exceptions.py
```

---

## 2. Key Technical Dependencies

### 2.1 Flask/Quart-Specific Dependencies

| Dependency | Purpose | Migration Difficulty |
|------------|---------|---------------------|
| `quart_cors` | CORS middleware | Medium — FastAPI has built-in `CORSMiddleware` |
| `quart_auth` | `Unauthorized` exception + `AuthUser` ORM mixin | Medium — see §2.1.1 |
| `quart_schema` | OpenAPI schema generation | Low — FastAPI has native support |
| `itsdangerous.URLSafeTimedSerializer` | Token minting/parsing (not JWT) | High — existing tokens incompatible with JWT swap |
| `flask_login` | Admin login | High — no direct FastAPI replacement |
| `flask_session` | Server-side sessions (admin) | High — no built-in FastAPI support |

### 2.1.1 Auth Architecture

The main API does **not** use `quart_auth`'s `login_required` or `current_user`. Those are custom-built in `api/apps/__init__.py:96–189`:

- `_load_user()` — extracts `Authorization` header, deserialises via `URLSafeTimedSerializer`, falls back to `APIToken` lookup, sets `g.user`
- `current_user` — `LocalProxy(_load_user)`
- `login_required` — custom decorator wrapping `quart_auth.Unauthorized`

`quart_auth` is retained only for two things:
1. The `Unauthorized` exception class (imported in `__init__.py:31`)
2. The `AuthUser` mixin on the `User` ORM model (`db_models.py:707`: `class User(DataBaseModel, AuthUser)`)

**Migration task**: The `AuthUser` mixin embedded in the ORM model is a wider-scope change than just swapping API-layer imports. Removing it requires modifying `db_models.py` and auditing every call site that relies on `User` inheriting `AuthUser` behaviour.

### 2.1.2 Token Format

The login token is produced by `itsdangerous.URLSafeTimedSerializer` (`db_models.py:729`):

```python
def get_id(self):
    jwt = Serializer(secret_key=settings.get_secret_key())
    return jwt.dumps(str(self.access_token))
```

This is **not JWT**. The serialised blob wraps the `access_token` column value. Replacing this with `python-jose` or similar would invalidate all existing tokens at cutover. Either a dual-format backward-compatibility layer or a coordinated token-reissue window is required before any auth layer changes.

### 2.2 Flask Primitive Mappings

| Primitive | FastAPI Equivalent |
|-----------|-------------------|
| `request` (async) | `from fastapi import Request` |
| `jsonify()` | Return `dict` directly; FastAPI auto-serialises |
| `g.user` | `request.state.user` |
| `session` | Custom Redis session via `fastapi-redis-session` or equivalent |
| `Blueprint` | `fastapi.APIRouter` |
| `make_response` | Return `Response` object directly |
| `Response` (streaming) | `fastapi.responses.StreamingResponse` |

### 2.3 Dynamic Blueprint Loading

`api/apps/__init__.py:287–324` uses `importlib.util` to discover and load all `*_app.py`, `sdk/*.py`, and `restful_apis/*.py` files at startup, dynamically assigning each a `Blueprint` and registering it with the app. FastAPI's `APIRouter` is a direct equivalent for the registration mechanism, but the `importlib`-based discovery loop would need to be reproduced or replaced with explicit router imports.

### 2.4 Custom JSON Encoder

```python
class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date, timedelta)): ...
        if isinstance(o, Enum): return o.value
        if isinstance(o, set): return list(o)
        if isinstance(o, BaseType): return str(o)
```

Reproducible in FastAPI via a custom `ORJSONResponse` or overriding `json.dumps`.

---

## 3. Migration Complexity

### 3.1 High-Risk Areas

| Module | Reason |
|--------|--------|
| `api/apps/__init__.py` (367 lines) | App factory, custom auth middleware, dynamic blueprint loading, error handlers |
| `api/utils/api_utils.py` (796 lines) | `get_json_result()` and helpers depend on Quart request context |
| `api/db/db_models.py` | `AuthUser` mixin on `User` model; `URLSafeTimedSerializer` token format |
| `admin/server/` | Flask + `flask_login` + `flask_session`; no direct FastAPI equivalent |

### 3.2 Medium-Risk Areas

| Module | Reason |
|--------|--------|
| `restful_apis/agent_api.py` (~1,892 lines) | Large file; logic is mostly framework-independent but volume is high |
| `restful_apis/document_api.py` (~1,895 lines) | Same |
| `restful_apis/file_api.py` | Multipart/form upload handling differs between Quart and FastAPI |
| `restful_apis/connector_api.py` | OAuth flows with session-backed state; HTML response construction |
| `restful_apis/user_api.py` | OAuth callback handling |

### 3.3 Route Complexity Breakdown

| Category | Files | Complexity |
|----------|-------|------------|
| Simple JSON CRUD | ~8–10 files (stats, plugin, task, etc.) | Low — decorator swap |
| Multipart/file uploads | `file_api.py` | Medium — `request.files`/`request.form` differ |
| OAuth flows + session state | `user_api.py`, `connector_api.py` | High — session-backed state |
| Custom non-JSON responses | `connector_api.py` (`make_response(html, 200)`) | Medium — `HTMLResponse` |
| Mixed query args + JSON body | `mcp_api.py`, `memory_api.py` | Medium |

### 3.4 Validation Layer — Already FastAPI-Compatible

`api/utils/validation_utils.py` (1,043 lines) uses Pydantic v2 with `ConfigDict(extra="forbid", strict=True)` on the base model class. The current API already enforces strict validation. Migration must preserve this behaviour — it should not be weakened.

---

## 4. Phased Workload Estimate

| Phase | Content | Complexity |
|-------|---------|------------|
| Phase 0 | FastAPI skeleton + auth middleware + token strategy | High |
| Phase 1 | Rewrite `__init__.py` + `api_utils.py` + remove `AuthUser` mixin | High |
| Phase 2A | Simple JSON CRUD routes (~8–10 files) | Medium |
| Phase 2B | Complex routes (uploads, OAuth, HTML, session-backed) | High |
| Phase 3 | Admin panel (optional; keep on Flask with separate deployment) | High |
| Phase 4 | Testing + regression verification | Medium |

**Estimated total**: 5–7 person-months, depending on token migration strategy.

Phase 0 is the hard prerequisite. The token format decision blocks all downstream work.

---

## 5. FastAPI Benefits vs. Current Stack

| Benefit | FastAPI | Current Stack |
|---------|---------|---------------|
| Native OpenAPI generation | Yes | Yes — via `quart_schema` |
| Pydantic v2 validation | Yes | Yes — `validation_utils.py` already uses Pydantic v2 |
| Full async support | Yes | Yes — Quart is fully async |
| Dependency injection | Yes | Partial — custom decorators |
| Larger ecosystem/community | Yes | — |
| Type-annotated route params | Yes | Partial |

The three headline FastAPI advantages (OpenAPI, Pydantic v2, async) are already present in the current stack. The net incremental gain is dependency injection style and ecosystem momentum.

---

## 6. Decision

**Status: Deferred.**

### Rationale

1. **Benefits already present.** The current stack already delivers OpenAPI generation (`quart_schema`), Pydantic v2 strict validation (`validation_utils.py`), and full async via Quart. The incremental value of FastAPI does not justify the disruption at this time.

2. **Token format is a hard blocker.** The `URLSafeTimedSerializer` token format is incompatible with JWT. Any auth layer migration requires either maintaining the serialiser in FastAPI or coordinating a token-reissue rollout. This decision must be made and executed before any routes can be migrated. Without resolving it first, a partial migration would leave the auth system in an inconsistent state.

3. **`AuthUser` mixin scope is wider than it appears.** `quart_auth.AuthUser` is embedded in the `User` ORM model (`db_models.py`), not just the API layer. Removing it touches the data model and all downstream code that depends on `User`, making the change harder to isolate and test.

4. **Ongoing asyncio correctness work.** Concurrent framework-level changes compound risk and make it harder to isolate regressions.

5. **Admin panel has no clean FastAPI path.** `flask_login` and `flask_session` have no direct FastAPI equivalents. The admin panel would need to remain on Flask and be deployed separately, splitting the stack rather than unifying it.

### Conditions for Revisiting

This decision should be revisited if:

- A concrete pain point emerges that Quart demonstrably cannot address (e.g., a specific library that only integrates with FastAPI's `Depends` system)
- The `URLSafeTimedSerializer` token format is replaced for independent reasons (security audit, token expiry policy change), making the auth migration a free side-effect
- The admin panel is retired or rebuilt independently, removing the `flask_login`/`flask_session` constraint

### If Migration Is Pursued in Future

Start with Phase 0 only: build the FastAPI skeleton, port the auth layer, and decide the token strategy — without touching any route files. Validate that approach as a standalone PR before committing to Phase 1 onward.
