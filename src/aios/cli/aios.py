import typer
from typing import Optional

app = typer.Typer(name="aios", help="AI-OS Command Line Interface")

@app.command()
def gui():
    """Launch the AI-OS GUI application."""
    from aios.gui.main import main
    main()

@app.command()
def version():
    """Show the AI-OS version."""
    typer.echo("AI-OS version 1.0.0")

if __name__ == "__main__":
    app()
