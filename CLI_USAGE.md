# KMNLP API CLI

A command-line interface for interacting with the KMNLP API. The CLI provides both interactive chat sessions and direct API commands for testing and automation.

## Quick Start

```bash
# Install and setup
source .venv/bin/activate
uv sync
uv pip install -e .

# Start the API server in background
export AWS_PROFILE=kmnlp
fastapi dev app/main.py &

# Start an interactive chat session (recommended)
kmnlp chat start

# Or use direct API commands
kmnlp users create my-user
kmnlp --user-id my-user chats create
```

## Installation

The CLI is automatically available when you install the project dependencies:

```bash
source .venv/bin/activate
uv sync
uv pip install -e .  # Install in development mode to enable 'kmnlp' command
```

## Usage

### Basic Command Structure

```bash
kmnlp [OPTIONS] COMMAND [ARGS]...
```

### Interactive Chat Commands (Recommended)

```bash
# Start an interactive chat session
kmnlp chat start

# List your chat sessions
kmnlp chat list
```

### Direct API Commands

For automation or testing, you can use the low-level API commands:

### Global Options

- `--base-url TEXT`: Base URL for the API (default: http://127.0.0.1:8000)
- `--user-id TEXT`: User ID for authenticated requests

### User Management

```bash
# List all users
kmnlp users list

# Create a new user
kmnlp users create <user_id>

# Get user details
kmnlp users get <user_id>
```

### Chat Management

```bash
# List chats for a user (requires --user-id)
kmnlp --user-id <user_id> chats list

# Create a new chat (requires --user-id)
kmnlp --user-id <user_id> chats create

# Get chat details (requires --user-id)
kmnlp --user-id <user_id> chats get <chat_id>

# Send a message to a chat (requires --user-id)
kmnlp --user-id <user_id> chats send <chat_id> "<message>"

# Send a message with stub GeoTIFF (for testing)
kmnlp --user-id <user_id> chats send <chat_id> "<message>" --stub-geotiff
```

## Examples

### Interactive Chat Session Example (Recommended)

```bash
# Set environment variable (API server runs on 127.0.0.1:8000 by default)
export AWS_PROFILE=kmnlp

# Start an interactive chat session
kmnlp chat start

# This will:
# 1. Automatically create a user if needed
# 2. Create or resume a chat session
# 3. Allow you to type messages directly
# 4. Display streaming responses in real-time
# 5. Support commands like /help, /new, /history, /quit
```

In the interactive session, you can:
- Type any message and get a response
- Use `/help` to see available commands
- Use `/new` to start a new chat
- Use `/history` to see recent messages
- Use `/quit` to exit

### Complete Workflow Example (Direct API Commands)

```bash
# Set environment variable (API server runs on 127.0.0.1:8000 by default)
export AWS_PROFILE=kmnlp

# 1. Create a user
kmnlp users create test-user

# 2. Verify user was created
kmnlp users get test-user

# 3. Create a chat for the user
kmnlp --user-id test-user chats create

# 4. List chats for the user (should show the new chat ID)
kmnlp --user-id test-user chats list

# 5. Send a message (replace <chat_id> with actual chat ID from step 4)
kmnlp --user-id test-user chats send <chat_id> "Hello, can you help me find weather data?"

# 6. Check chat details to see message history
kmnlp --user-id test-user chats get <chat_id>
```

## Environment Setup

Before using the CLI with the full LLM functionality, ensure you have the AWS profile configured:

```bash
export AWS_PROFILE=kmnlp
```

This is required for the Bedrock API integration to work properly.

## Response Formats

### Interactive Mode
In interactive chat sessions, responses are displayed in real-time with:
- **Text responses**: Streamed as they arrive from the LLM
- **Plans**: Displayed with step-by-step progress indicators (⏳ pending, 🔄 in progress, ✅ success, ❌ failure)
- **Special commands**: `/help`, `/new`, `/history`, `/quit`
- **Error handling**: Graceful display of connection and API errors

### Direct API Mode
The CLI displays responses in a user-friendly format:

- **Text responses**: Displayed as plain text
- **Plans**: Displayed with step-by-step progress indicators (⏳ pending, 🔄 in progress, ✅ success, ❌ failure)
- **GeoTIFF**: Displays the URL of the map data
- **Errors**: Displayed with ❌ prefix

## Notes

- The API server must be running for the CLI to work (use `fastapi dev app/main.py`)
- **Interactive mode** is the recommended way to use the CLI for most users
- The CLI defaults to connecting to `http://127.0.0.1:8000` (where the dev server runs)
- Use `--base-url` option to connect to a different server if needed
- Interactive sessions automatically handle user creation and session management
- Session configuration is saved in `~/.config/kmnlp/config.json`
- Streaming responses may timeout if AWS credentials are not properly configured
- The CLI automatically handles JSON serialization of datetime fields
- All HTTP errors are caught and displayed with appropriate error messages
- Use `kmnlp --help` or `kmnlp <command> --help` for detailed help information
