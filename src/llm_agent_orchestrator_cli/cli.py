import click
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

@click.group()
def cli():
    """LLM Agent Orchestrator CLI"""
    pass

@cli.command()
@click.option('--name', default='World', help='Name to greet.')
def hello(name):
    """Says hello to the given name."""
    click.echo(f"Hello, {name}!")
    # Exemplo de como acessar uma variável de ambiente (se definida)
    # api_key = os.getenv("OPENAI_API_KEY")
    # if api_key:
    #     click.echo("OpenAI API Key is loaded.")
    # else:
    #     click.echo("OpenAI API Key not found in environment.")

if __name__ == '__main__':
    cli()
