# ruff: noqa: C901, PLR0912, PLR0915

"""Main CLI implementation for KMNLP API."""

import asyncio
import json
import logging
import sys
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

# Langfuse environment variables must be loaded earlier than this
load_dotenv()

from app.types.input import UserCreateRequest  # noqa: E402
from app.types.output import (  # noqa: E402
    BackendError,
    Chat,
    ChatRequest,
    ChatResponse,
    GeocodedAOI,
    GeoTIFF,
    Image,
    Plan,
    Text,
    UserResponse,
)

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)


class CLISession:
    """Manages CLI session state including user and config persistence."""

    def __init__(self, config_dir: Path | None = None) -> None:
        """Initialize CLI session.

        Args:
            config_dir: Directory to store config files. Defaults to ~/.kmnlp/
        """
        self.config_dir = config_dir or Path.home() / ".kmnlp"
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self._config: dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with self.config_file.open() as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._config = {}

    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            with self.config_file.open("w") as f:
                json.dump(self._config, f, indent=2)
        except OSError:
            pass  # Fail silently if we can't save config

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = {}
        self._save_config()

    @property
    def user_id(self) -> str | None:
        """Get the current user ID."""
        return self._config.get("user_id")

    @user_id.setter
    def user_id(self, value: str) -> None:
        """Set the current user ID."""
        self._config["user_id"] = value
        self._save_config()

    @property
    def last_chat_id(self) -> str | None:
        """Get the last used chat ID."""
        return self._config.get("last_chat_id")

    @last_chat_id.setter
    def last_chat_id(self, value: str) -> None:
        """Set the last used chat ID."""
        self._config["last_chat_id"] = value
        self._save_config()

    @property
    def base_url(self) -> str:
        """Get the API base URL."""
        return self._config.get("base_url", "http://127.0.0.1:8000")

    @base_url.setter
    def base_url(self, value: str) -> None:
        """Set the API base URL."""
        self._config["base_url"] = value
        self._save_config()


class APIClient:
    """HTTP client for KMNLP API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", user_id: str | None = None) -> None:
        """Initialize API client.

        Args:
            base_url: Base URL for the API
            user_id: User ID for authenticated requests
        """
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        # Use longer timeout for LLM streaming responses
        self.client = httpx.AsyncClient(timeout=300.0)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.user_id:
            headers["x-user-id"] = self.user_id
        return headers

    async def create_user(self, user_id: str) -> str:
        """Create a new user."""
        response = await self.client.post(
            f"{self.base_url}/users",
            json=UserCreateRequest(user_id=user_id).model_dump(),
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.text.strip('"')

    async def get_users(self) -> list[str]:
        """Get all users."""
        response = await self.client.get(
            f"{self.base_url}/users",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def get_user(self, user_id: str) -> UserResponse:
        """Get user details."""
        response = await self.client.get(
            f"{self.base_url}/users/{user_id}",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return UserResponse.model_validate(response.json())

    async def get_user_chats(self) -> list[str]:
        """Get chats for the current user."""
        if not self.user_id:
            msg = "User ID is required for this operation"
            raise ValueError(msg)
        response = await self.client.get(
            f"{self.base_url}/chats",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def create_chat(self) -> Chat:
        """Create a new chat."""
        if not self.user_id:
            msg = "User ID is required for this operation"
            raise ValueError(msg)
        response = await self.client.post(
            f"{self.base_url}/chats",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return Chat.model_validate(response.json())

    async def get_chat(self, chat_id: str) -> Chat:
        """Get chat details."""
        if not self.user_id:
            msg = "User ID is required for this operation"
            raise ValueError(msg)
        response = await self.client.get(
            f"{self.base_url}/chats/{chat_id}",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return Chat.model_validate(response.json())

    async def send_message(
        self, chat_id: str, message: str, *, stub_geotiff: bool = False
    ) -> AsyncGenerator[ChatResponse, None]:
        """Send a message to a chat and stream responses."""
        if not self.user_id:
            msg = "User ID is required for this operation"
            raise ValueError(msg)

        chat_request = ChatRequest(content=message)
        params = {"stub_geotiff": "true"} if stub_geotiff else {}

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chats/{chat_id}",
            headers=self._get_headers(),
            json=chat_request.model_dump(mode="json"),
            params=params,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        chat_response = ChatResponse.model_validate(data)
                        yield chat_response

                        # Stop streaming when we receive the end-turn signal
                        content_type = getattr(chat_response.content, "content_type", None)
                        if content_type == "EndOfTurn":
                            break
                    except (json.JSONDecodeError, ValidationError) as e:
                        click.echo(f"Error parsing response: {e}", err=True)
                        continue

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


async def _ensure_user_exists(client: APIClient, session: CLISession) -> None:
    """Ensure a user exists and is set in the session."""
    if not session.user_id:
        user_id = f"user-{uuid.uuid4().hex[:8]}"
        await client.create_user(user_id)
        session.user_id = user_id
        click.echo(f"Created new user: {user_id}")
    client.user_id = session.user_id


async def _get_or_create_chat(client: APIClient, session: CLISession, *, new: bool) -> str:
    """Get existing chat or create a new one."""
    chat_id = None

    if not new and session.last_chat_id:
        try:
            await client.get_chat(session.last_chat_id)
            chat_id = session.last_chat_id
            click.echo(f"Resuming chat session: {chat_id}")
        except httpx.HTTPStatusError:
            pass  # Last chat doesn't exist anymore

    if not chat_id:
        chat = await client.create_chat()
        chat_id = chat.id
        session.last_chat_id = chat_id
        click.echo(f"Started new chat session: {chat_id}")

    return chat_id


def _handle_chat_command(user_input: str) -> str | None:
    """Handle special chat commands.

    Returns:
        Command to execute, or None if not a command
    """
    command_map = {
        "/quit": "quit",
        "/exit": "quit",
        "/q": "quit",
        "/help": "help",
        "/new": "new",
        "/history": "history",
    }
    return command_map.get(user_input.strip().lower())


@click.group()
@click.option(
    "--base-url",
    default="http://127.0.0.1:8000",
    help="Base URL for the API",
    show_default=True,
)
@click.option("--user-id", help="User ID for authenticated requests")
@click.pass_context
def cli(ctx: click.Context, base_url: str, user_id: str | None) -> None:
    """KMNLP Chat Assistant - Natural Language Interface for NOAA Data."""
    ctx.ensure_object(dict)
    session = CLISession()

    # Override session values with command line arguments
    if base_url != "http://127.0.0.1:8000":
        session.base_url = base_url
    if user_id:
        session.user_id = user_id

    ctx.obj["session"] = session
    ctx.obj["client"] = APIClient(base_url=session.base_url, user_id=session.user_id)


@cli.group()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Interactive chat commands."""


@chat.command("start")
@click.option(
    "--new",
    is_flag=True,
    help="Start a new chat session (don't resume the last one)",
)
@click.pass_context
def chat_start(ctx: click.Context, *, new: bool = False) -> None:
    """Start an interactive chat session."""

    async def _chat_start() -> None:
        session: CLISession = ctx.obj["session"]
        client: APIClient = ctx.obj["client"]

        try:
            await _ensure_user_exists(client, session)
            chat_id = await _get_or_create_chat(client, session, new=new)

            click.echo("\n" + "=" * 60)
            click.echo("🤖 KMNLP Chat Assistant")
            click.echo("Type your questions about NOAA data and weather information.")
            click.echo("Commands: /help, /quit, /new, /history")
            click.echo("=" * 60 + "\n")

            # Interactive chat loop
            while True:
                try:
                    user_input = click.prompt("You", prompt_suffix=" > ", show_default=False)

                    command = _handle_chat_command(user_input)
                    if command == "quit":
                        click.echo("Goodbye! 👋")
                        break
                    if command == "help":
                        _show_help()
                        continue
                    if command == "new":
                        chat = await client.create_chat()
                        chat_id = chat.id
                        session.last_chat_id = chat_id
                        click.echo(f"Started new chat session: {chat_id}\n")
                        continue
                    if command == "history":
                        await _show_history(client, chat_id)
                        continue
                    if not user_input.strip():
                        continue

                    # Send message and stream response
                    click.echo("\n🤖 Assistant:")
                    click.echo("-" * 40)

                    async for response in client.send_message(chat_id, user_input):
                        _display_chat_response(response)

                    click.echo("\n")

                except (EOFError, KeyboardInterrupt):
                    click.echo("\nGoodbye! 👋")
                    break
                except httpx.HTTPStatusError as e:
                    try:
                        detail = e.response.json()["detail"]
                    except (KeyError, json.JSONDecodeError):
                        detail = str(e)
                    click.echo(f"\n❌ API Error: {detail}", err=True)
                    click.echo("Please try again.\n")

        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()["detail"]
            except (KeyError, json.JSONDecodeError):
                detail = str(e)
            click.echo(f"❌ API Error: {detail}", err=True)
            sys.exit(1)
        except (ValueError, OSError) as e:
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_chat_start())


def _show_help() -> None:
    """Show help information for the interactive chat."""
    click.echo("\n" + "=" * 50)
    click.echo("💡 Chat Commands:")
    click.echo("  /help     - Show this help message")
    click.echo("  /quit     - Exit the chat session")
    click.echo("  /new      - Start a new chat conversation")
    click.echo("  /history  - Show recent chat history")
    click.echo("\n💬 Example Questions:")
    click.echo("  • What's the weather like in Miami?")
    click.echo("  • Show me sea surface temperature data for the Gulf of Mexico")
    click.echo("  • Find precipitation data for California last month")
    click.echo("=" * 50 + "\n")


async def _show_history(client: APIClient, chat_id: str) -> None:
    """Show recent chat history."""
    try:
        chat = await client.get_chat(chat_id)
        if not chat.messages:
            click.echo("No messages in this chat yet.\n")
            return

        click.echo("\n" + "=" * 50)
        click.echo("📜 Recent Chat History:")
        click.echo("=" * 50)

        # Show last 10 messages
        for i, message in enumerate(chat.messages[-10:], 1):
            if hasattr(message, "content") and isinstance(message.content, str):
                # User message
                click.echo(f"\n{i}. 👤 You: {message.content}")
            else:
                # Assistant response (just show a summary)
                content_type = getattr(message.content, "content_type", "Unknown")
                click.echo(f"{i}. 🤖 Assistant: [{content_type} response]")

        click.echo("=" * 50 + "\n")
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json()["detail"]
        except (KeyError, json.JSONDecodeError):
            detail = "Unknown error"
        click.echo(f"❌ Error loading history: {detail}\n")


@chat.command("list")
@click.pass_context
def chat_list(ctx: click.Context) -> None:
    """List your recent chat sessions."""

    async def _chat_list() -> None:
        session: CLISession = ctx.obj["session"]
        client: APIClient = ctx.obj["client"]

        if not session.user_id:
            click.echo("No user session found. Run 'kmnlp chat start' first.")
            return

        client.user_id = session.user_id

        try:
            chat_ids = await client.get_user_chats()
            if not chat_ids:
                click.echo("No chat sessions found.")
                return

            click.echo("Your chat sessions:")
            for i, chat_id in enumerate(chat_ids, 1):
                try:
                    chat = await client.get_chat(chat_id)
                    msg_count = len(chat.messages)
                    status = " (current)" if chat_id == session.last_chat_id else ""
                    click.echo(f"  {i}. {chat_id} - {msg_count} messages{status}")
                except httpx.HTTPStatusError:
                    click.echo(f"  {i}. {chat_id} - (error loading details)")

        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()["detail"]
            except (KeyError, json.JSONDecodeError):
                detail = str(e)
            click.echo(f"❌ Error: {detail}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_chat_list())


@cli.command("config")
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--reset", is_flag=True, help="Reset configuration to defaults")
@click.pass_context
def config_cmd(ctx: click.Context, *, show: bool = False, reset: bool = False) -> None:
    """Manage CLI configuration."""
    session: CLISession = ctx.obj["session"]

    if reset:
        session.reset_config()
        click.echo("Configuration reset to defaults.")
        return

    if show:
        click.echo("Current configuration:")
        click.echo(f"  User ID: {session.user_id or '(not set)'}")
        click.echo(f"  API URL: {session.base_url}")
        click.echo(f"  Last Chat: {session.last_chat_id or '(none)'}")
        click.echo(f"  Config file: {session.config_file}")
        return

    click.echo("Use --show to display current configuration or --reset to clear it.")


# Legacy command group for backwards compatibility (hidden from help)
@cli.group(hidden=True)
@click.pass_context
def users(ctx: click.Context) -> None:
    """User management commands."""


@users.command("create")
@click.argument("user_id")
@click.pass_context
def create_user(ctx: click.Context, user_id: str) -> None:
    """Create a new user."""

    async def _create_user() -> None:
        client: APIClient = ctx.obj["client"]
        try:
            created_user_id = await client.create_user(user_id)
            click.echo(f"Created user: {created_user_id}")
        except httpx.HTTPStatusError as e:
            click.echo(f"Error creating user: {e.response.json()['detail']}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_create_user())


@users.command("list")
@click.pass_context
def list_users(ctx: click.Context) -> None:
    """List all users."""

    async def _list_users() -> None:
        client: APIClient = ctx.obj["client"]
        try:
            user_list = await client.get_users()
            if user_list:
                for user in user_list:
                    click.echo(user)
            else:
                click.echo("No users found")
        except httpx.HTTPStatusError as e:
            click.echo(f"Error listing users: {e.response.json()['detail']}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_list_users())


@users.command("get")
@click.argument("user_id")
@click.pass_context
def get_user(ctx: click.Context, user_id: str) -> None:
    """Get user details."""

    async def _get_user() -> None:
        client: APIClient = ctx.obj["client"]
        try:
            user_response = await client.get_user(user_id)
            click.echo(f"User ID: {user_response.user_id}")
            chat_ids_display = (
                ", ".join(user_response.chat_ids) if user_response.chat_ids else "None"
            )
            click.echo(f"Chat IDs: {chat_ids_display}")
        except httpx.HTTPStatusError as e:
            click.echo(f"Error getting user: {e.response.json()['detail']}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_get_user())


@cli.group()
@click.pass_context
def chats(ctx: click.Context) -> None:
    """Chat management commands."""


@chats.command("list")
@click.pass_context
def list_chats(ctx: click.Context) -> None:
    """List chats for the current user."""

    async def _list_chats() -> None:
        client: APIClient = ctx.obj["client"]
        try:
            chat_list = await client.get_user_chats()
            if chat_list:
                for chat_id in chat_list:
                    click.echo(chat_id)
            else:
                click.echo("No chats found")
        except httpx.HTTPStatusError as e:
            click.echo(f"Error listing chats: {e.response.json()['detail']}", err=True)
            sys.exit(1)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_list_chats())


@chats.command("create")
@click.pass_context
def create_chat(ctx: click.Context) -> None:
    """Create a new chat."""

    async def _create_chat() -> None:
        client: APIClient = ctx.obj["client"]
        try:
            chat = await client.create_chat()
            click.echo(f"Created chat: {chat.id}")
        except httpx.HTTPStatusError as e:
            click.echo(f"Error creating chat: {e.response.json()['detail']}", err=True)
            sys.exit(1)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_create_chat())


@chats.command("get")
@click.argument("chat_id")
@click.pass_context
def get_chat(ctx: click.Context, chat_id: str) -> None:
    """Get chat details."""

    async def _get_chat() -> None:
        client: APIClient = ctx.obj["client"]
        try:
            chat = await client.get_chat(chat_id)
            click.echo(f"Chat ID: {chat.id}")
            click.echo(f"Messages: {len(chat.messages)}")
            if chat.messages:
                click.echo("\nRecent messages:")
                for i, message in enumerate(chat.messages[-3:], 1):
                    if hasattr(message, "content") and isinstance(message.content, str):
                        # ChatRequest
                        click.echo(f"  {i}. User: {message.content[:100]}...")
                    else:
                        # ChatResponse
                        click.echo(f"  {i}. Assistant: {type(message.content).__name__}")
        except httpx.HTTPStatusError as e:
            click.echo(f"Error getting chat: {e.response.json()['detail']}", err=True)
            sys.exit(1)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_get_chat())


@chats.command("send")
@click.argument("chat_id")
@click.argument("message")
@click.option("--stub-geotiff", is_flag=True, help="Include a stub GeoTIFF in the response")
@click.pass_context
def send_message(
    ctx: click.Context, chat_id: str, message: str, *, stub_geotiff: bool = False
) -> None:
    """Send a message to a chat."""

    async def _send_message() -> None:
        client: APIClient = ctx.obj["client"]
        try:
            click.echo(f"Sending message to chat {chat_id}...")
            click.echo(f"Message: {message}")
            click.echo("=" * 50)

            async for response in client.send_message(chat_id, message, stub_geotiff=stub_geotiff):
                _display_chat_response(response)

        except httpx.HTTPStatusError as e:
            click.echo(f"Error sending message: {e.response.json()['detail']}", err=True)
            sys.exit(1)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_send_message())


def _display_text_content(content: Text) -> None:
    """Display text content."""
    click.secho(content.text, nl=False, fg="yellow")


def _display_plan_content(content: Plan) -> None:
    """Display plan content."""
    click.echo("-" * 80)
    click.secho(f"\n📋 Plan: {content.title}", fg="magenta", bold=True)
    for step in content.steps:
        status_emoji = {
            "pending": "⏳",
            "in_progress": "🔄",
            "success": "✅",
            "failure": "❌",
            "cancelled": "🚫",
        }.get(step.status, "❓")
        click.secho(f"  {status_emoji} {step.title}", fg="magenta")
        if step.outcome_summary:
            click.secho(f"    → {step.outcome_summary}", fg="magenta")

        # Display nested progress updates within the step
        if step.progress_updates:
            for progress_update in step.progress_updates:
                click.secho(f"    ⚡ {progress_update.title}", fg="cyan", dim=True)
    click.echo("-" * 80)


def _display_geotiff_content(content: GeoTIFF) -> None:
    """Display GeoTIFF content."""
    click.echo(f"\n🗺️  GeoTIFF: {content.url}")


def _display_error_content(content: BackendError) -> None:
    """Display error content."""
    click.echo(f"\n❌ Error: {content.message}", err=True)


def _display_geocoded_aoi_content(content: GeocodedAOI) -> None:
    """Display geocoded AOI content."""
    click.secho(f"\n🌍 Geocoded AOI for: {content.query}", fg="green", bold=True)
    click.secho(f"📍 Found {content.geojson!r}", fg="green")


def _display_image_content(content: Image) -> None:
    """Display image content."""
    click.secho(f"\n🖼️  Generated image: {content.media_type}", fg="green", bold=True)
    click.secho(f"Image URL: {content.url}", fg="green")


def _display_chat_response(response: ChatResponse) -> None:
    """Display a chat response in a user-friendly format."""
    content = response.content

    # Handle GeocodedAOI specifically since it may have different structure
    if isinstance(content, GeocodedAOI):
        _display_geocoded_aoi_content(content)
        return

    # For other types, use content_type
    content_type = getattr(content, "content_type", type(content).__name__)

    match content_type:
        case "EndOfTurn":
            # End of turn - don't display anything, just let the loop terminate
            pass
        case "Text":
            _display_text_content(Text.model_validate(content))
        case "Plan":
            _display_plan_content(Plan.model_validate(content))
        case "GeoTIFF":
            _display_geotiff_content(GeoTIFF.model_validate(content))
        case "Image":
            _display_image_content(Image.model_validate(content))
        case "BackendError":
            _display_error_content(BackendError.model_validate(content))
        case _:
            click.echo(f"\n🔧 {content_type}: {content}")


if __name__ == "__main__":
    cli()
