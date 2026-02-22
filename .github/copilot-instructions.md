# KMNLP API - Copilot Instructions

## Architecture Overview

This is a FastAPI-based chat service that provides streaming responses via LLM agents for NOAA Knowledge Management NLP. The service implements a multi-user chat system with SQLite persistence and integrates with the `llm-agent` private dependency for orchestration.

### Key Components

- **FastAPI app** (`app/main.py`): RESTful chat API with CORS, user management, and streaming responses
- **Database layer** (`app/db/`): SQLite-based persistence with Protocol interface for user/chat storage
- **Agent orchestration**: Uses `OrchestrationAgent` from private `llm-agent` library for data search/analysis
- **Type system** (`app/types.py`): Comprehensive Pydantic models with discriminated unions for chat content
- **Utilities** (`app/utils/`): Agent factories, response streaming, and plan management

## Development Workflows

### Local Development

```bash
# FIRST: Activate virtual environment (required for all commands below)
source .venv/bin/activate

# Start development server (hot reload enabled)
fastapi dev app/main.py

# Run all linting checks
scripts/lint.sh  # includes shellcheck, ruff, pyright, format check

# Run tests with verbose output
scripts/test.sh  # pytest with -vv -rA --log-cli-level=INFO

# Run integration tests separately
scripts/integration_tests.sh  # pytest -m integration with verbose logging

# Build everything (package + docker images)
scripts/build.sh  # builds both api and dask containers
```

### Testing Patterns

- **Integration tests**: Marked with `@pytest.mark.integration`, require `-m integration` flag
- **Default behavior**: Only unit tests run unless explicitly marked for integration
- **Test structure**: `tests/unit/` and `tests/integration/` mirrors `app/` structure
- **API testing**: Uses `TestClient` from FastAPI for endpoint testing

## Dependency Management

### Python Environment

- **Package manager**: `uv` (modern pip replacement)
- **Python version**: 3.12+ required
- **Virtual environment**: **ALWAYS** activate with `source .venv/bin/activate` before running any scripts or commands
- **Private dependency**: `llm-agent` from GitLab SSH/HTTPS with CI tokens
- **Lock file**: `uv.lock` must be kept in sync with `pyproject.toml`

### Key Dependencies

- `fastapi[standard]>=0.116.1` - Web framework with built-in async support
- `llm-agent[data-processing]` - Private orchestration agent library
- `natural-language-geocoding` - Geocoding functionality for natural language queries
- `e84-geoai-common` - Shared LLM message types and content models (transitive dependency via llm-agent)

## Code Quality Standards

### Linting Configuration

- **Ruff**: Line length 100, Google docstring convention, ALL rules with specific ignores
- **Pyright**: Strict type checking with comprehensive error reporting
- **Pre-commit**: Hooks for automated quality checks

### Critical Ignores

```python
# In tests/: Allow assert statements (S101), missing docstrings (D1)
# In __init__.py: Allow unused imports (F401) for re-exports
# Project-wide: No TO D O author requirements (TD002/TD003)
```

## API Patterns

### Authentication

- **Header-based**: `x-user-id` header required for all authenticated endpoints
- **User management**: In-memory user storage with UUID-based chat sessions
- **Error handling**: Custom HTTPException subclasses (`MissingUserIdError`, etc.)

### Streaming Responses

```python
# Pattern for chat endpoints
@app.post("/chat/{chat_id}", response_class=StreamingResponse)
async def post_to_chat(...) -> StreamingResponse:
    # Convert to LLMMessage with TextContent + CachePointContent
    # Stream via get_agent_response_content_stream()
    # Return application/x-ndjson with ChatResponse objects
```

### Type Safety

- **Discriminated unions**: `chat_response_content_types` uses `content_type` discriminator
- **Pydantic validation**: All request/response models with comprehensive field validation

## Container Architecture

### Multi-stage Build

```dockerfile
# base: Python + uv + dependencies
# build: Package building stage
# api: FastAPI service (port 8000)
# dask: Distributed computing image
```

### CI/CD Integration

- **Local builds**: Use SSH keys for private repo access
- **CI builds**: Use GitLab CI tokens with HTTPS git URL rewriting
- **Environment detection**: `BUILD_ENV_IS_CI` flag controls auth method

## Agent Integration

### Tool Integration

- **Data search**: `data_search_tool_with_tracking`
- **Data analysis**: `data_analysis_tool_with_tracking`
- **Response shaping**: `agent_plan_tool`, `text_response_tool`, `end_turn_tool`
- **Plan management**: Custom `PlanManager` for step tracking

## File Organization

### Core Structure

```
app/
├── main.py          # FastAPI app, routes, middleware
├── types.py         # Pydantic models, type definitions
├── db/
│   ├── __init__.py  # Database module exports
│   ├── interface.py # UserDatabase Protocol interface
│   └── sqlite_db.py # SQLite implementation with user/chat tables
└── utils/
    ├── utils.py     # Agent factories, streaming utilities
    └── plan_manager.py  # Plan/step management logic
```

### Testing Mirror

```
tests/
├── conftest.py      # Pytest configuration, integration marker handling
├── unit/app/        # Mirrors app/ structure for unit tests
└── integration/app/ # Integration tests requiring network calls
```

## Common Gotchas

1. **Integration tests**: Default pytest run skips them; use `-m integration` explicitly
2. **Private repo access**: Ensure SSH keys configured locally or CI tokens in GitLab
3. **UV lock sync**: Run `uv lock --check` to validate lock file consistency
4. **Type checking**: Pyright strict mode catches many runtime issues early
5. **Response streaming**: Use `application/x-ndjson` for chat streaming, not regular JSON

## Code Quality Requirements

**CRITICAL**: Before declaring any code changes complete, you MUST run and resolve ALL of the following:

### Mandatory Quality Checks

```bash
# FIRST: Activate virtual environment (required for all commands below)
source .venv/bin/activate

# 1. Fix all linting errors
scripts/lint.sh

# 2. Run tests to ensure functionality
scripts/test.sh

# 3. Format code properly
ruff format app/ tests/
```

### Quality Standards

- **Zero tolerance**: No ruff linting errors, pyright type errors, or shellcheck issues
- **Type safety**: All functions must have proper type annotations and pass strict pyright checks
- **Import organization**: Follow PEP 8 import ordering, remove unused imports
- **Docstrings**: All public functions/classes must have Google-style docstrings
- **Error handling**: Use proper exception chaining with `from err` or `from None`

### Before Completion Checklist

- [ ] `scripts/lint.sh` passes with no errors
- [ ] All imports are used and properly organized
- [ ] Type annotations are complete and accurate
- [ ] No trailing whitespace or formatting issues
- [ ] Tests pass (when applicable)
- [ ] Code follows established patterns in the codebase
