
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from ast import Dict

import click
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.text import Text
import os
import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from .rag_agent import RAGAgent
from .config import config

console = Console()

class CLI:
    def __init__(self):
        self.agent = None
        self.debug_mode = config.DEBUG_MODE
        
    def initialize(self, force_index: bool = False):
        """Initialize the RAG agent"""
        with console.status("[bold green]Initializing RAG Agent...[/bold green]"):
            # Set OpenAI API key
            if config.OPENAI_API_KEY:
                import openai
                openai.api_key = config.OPENAI_API_KEY
            else:
                console.print("[red]Error: OPENAI_API_KEY not set in environment[/red]")
                sys.exit(1)
                
            self.agent = RAGAgent(
                knowledge_base_path="./knowledge-base",
                orders_path="./data/orders.json",
                force_index=force_index
            )
            
        console.print("[green]✓[/green] Agent initialized successfully!")
        self._show_stats()
        
    def _show_stats(self):
        """Show agent statistics"""
        stats = self.agent.vector_store.get_stats()
        doc_summary = self.agent.doc_processor.get_document_summary()
        
        table = Table(title="Agent Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Documents", str(doc_summary['total_documents']))
        table.add_row("Document Chunks", str(stats['total_chunks']))
        
        table.add_row("Document Types", str(doc_summary.get('by_type', {})))
        
        console.print(table)
        
    def start_interactive(self):
        """Start interactive chat session"""
        self.agent.start_conversation()
        console.print(Panel(
            "[bold cyan]Aster & Row Support Agent[/bold cyan]\n"
            "Type your questions below. Type 'exit' to quit, 'debug' for debug info, 'clear' to clear conversation.",
            border_style="cyan"
        ))
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]")
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    console.print("[green]Goodbye! Feel free to reach out if you need more help.[/green]")
                    break
                    
                if user_input.lower() == 'clear':
                    self.agent.clear_conversation()
                    console.print("[yellow]Conversation cleared.[/yellow]")
                    continue
                    
                if user_input.lower() == 'debug':
                    self._show_debug_info(user_input)
                    continue
                    
                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                    
                # Process message
                with console.status("[bold blue]Thinking...[/bold blue]"):
                    result = self.agent.process_message(user_input)
                    
                # Display response
                self._display_response(result)
                
                # Show debug info if in debug mode
                if self.debug_mode:
                    self._show_debug_info(user_input)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                
    def _display_response(self, result: Dict):
        """Display agent response"""
        response = result['response']
        sources = result.get('sources', [])
        needs_human = result.get('needs_human_handoff', False)
        
        # Display response
        console.print("\n[bold cyan]Agent[/bold cyan]")
        console.print(Panel(Markdown(response), border_style="blue"))
        
        # Display sources
        if sources:
            source_text = "Sources: " + ", ".join([
                f"{s['filename']}" for s in sources
            ])
            console.print(f"[dim]{source_text}[/dim]")
            
        # Display handoff warning
        if needs_human:
            console.print("[yellow]⚠️ This may require human assistance. Please contact support if needed.[/yellow]")
            
        # Display order lookup status
        if result.get('order_lookup_performed'):
            console.print("[dim]Order lookup was performed.[/dim]")
            
        if result.get('conflicts_detected'):
            console.print("[yellow]⚠️ Multiple conflicting sources found. Human assistance recommended.[/yellow]")
            
        console.print("")
        
    def _show_debug_info(self, user_message: str):
        """Show debug information"""
        debug_info = self.agent.get_debug_info(user_message)
        
        console.print(Panel("[bold yellow]Debug Information[/bold yellow]", border_style="yellow"))
        
        # Retrieved documents
        if debug_info.get('retrieved_documents'):
            console.print("[bold]Retrieved Documents:[/bold]")
            for i, doc in enumerate(debug_info['retrieved_documents'], 1):
                console.print(f"  {i}. [dim]{doc['metadata'].get('filename')}[/dim] (score: {doc['score']:.3f})")
                console.print(f"     [dim]{doc['text'][:150]}...[/dim]")
                
        # Order data
        if debug_info.get('order_data'):
            console.print("[bold]Order Data:[/bold]")
            console.print(json.dumps(debug_info['order_data'], indent=2))
            
        # Conversation stats
        if debug_info.get('conversation_stats'):
            stats = debug_info['conversation_stats']
            console.print("[bold]Conversation Stats:[/bold]")
            console.print(f"  Total messages: {stats.get('total_messages', 0)}")
            console.print(f"  User messages: {stats.get('user_messages', 0)}")
            console.print(f"  Assistant messages: {stats.get('assistant_messages', 0)}")
            
    def run_evaluation(self, test_file: str = None):
        """Run evaluation suite"""
        import pytest
        import sys
        
        console.print("[bold]Running Evaluation Suite[/bold]")
        
        if test_file:
            args = ['-v', test_file]
        else:
            args = ['-v', 'tests/']
            
        sys.exit(pytest.main(args))

@click.group()
def main():
    """Aster & Row AI Support Agent"""
    pass

@main.command()
def chat():
    """Start interactive chat session"""
    cli = CLI()
    cli.initialize()
    cli.start_interactive()

@main.command()
def init():
    """Initialize the agent and index documents"""
    cli = CLI()
    cli.initialize(force_index=True)
    console.print("[green]✓[/green] Initialization complete!")

@main.command()
@click.option('--test-file', help='Specific test file to run')
def eval(test_file):
    """Run evaluation suite"""
    cli = CLI()
    cli.initialize()
    cli.run_evaluation(test_file)

@main.command()
@click.argument('query')
def ask(query):
    """Ask a single question"""
    cli = CLI()
    cli.initialize()
    cli.agent.start_conversation()
    result = cli.agent.process_message(query)
    cli._display_response(result)

if __name__ == '__main__':
    main()