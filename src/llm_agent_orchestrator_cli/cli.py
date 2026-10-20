import click
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

@click.group()
def main():
    """
    LLM Agent Orchestrator CLI.
    Orchestrates multiple LLMs and autonomous agents for complex tasks.
    """
    pass

@main.command()
def hello():
    """
    A simple hello command to test the CLI.
    """
    click.echo("Hello from LLM Agent Orchestrator CLI!")
    # Example of accessing an environment variable
    api_key = os.getenv("OPENAI_API_KEY", "Not Set")
    click.echo(f"OPENAI_API_KEY (from .env): {api_key[:5]}...") # Show first few chars
