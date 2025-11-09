"""
Optimization CLI commands for the unified optimization system.
"""

import typer
from pathlib import Path
from typing import Optional, List
import json

from aios.gui.components.hrm_training.optimizer_unified import optimize_cli, OptimizationConfig, optimize_from_config


def register(app: typer.Typer):
    """Register optimization commands with the main CLI app."""
    
    def _parse_device_list(value: str) -> List[int]:
        if not value:
            return []
        devices: List[int] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if token.lstrip("-").isdigit():
                try:
                    devices.append(int(token))
                except Exception:
                    continue
        return devices

    @app.command("optimize")
    def optimize_command(
        model: str = typer.Option("base_model", "--model", "-m", help="Model to optimize (HF model name or path)"),
        teacher: str = typer.Option("", "--teacher", help="Teacher model (defaults to same as model)"),
        max_seq: int = typer.Option(512, "--max-seq", help="Maximum sequence length"),
        test_duration: int = typer.Option(10, "--test-duration", help="Duration for each batch size test (seconds)"),
        max_timeout: int = typer.Option(20, "--max-timeout", help="Maximum subprocess timeout (seconds)"),
    batch_sizes: str = typer.Option("1,2,4,8,16,32", "--batch-sizes", help="Comma-separated list of batch sizes to test"),
        gen_samples: int = typer.Option(5, "--gen-samples", help="Number of samples for generation tests"),
        gen_tokens: int = typer.Option(8, "--gen-tokens", help="Max tokens per generation sample"),
        train_samples: int = typer.Option(3, "--train-samples", help="Number of samples for training tests"),
        train_tokens: int = typer.Option(4, "--train-tokens", help="Max tokens per training sample"),
        output_dir: str = typer.Option("artifacts/optimization", "--output-dir", "-o", help="Output directory for results"),
        cuda_devices: str = typer.Option("", "--cuda-devices", help="Comma-separated CUDA device IDs (e.g., '0,1')"),
        run_cuda_devices: str = typer.Option("", "--run-cuda-devices", help="CUDA devices for generation phase"),
        train_cuda_devices: str = typer.Option("", "--train-cuda-devices", help="CUDA devices for training phase"),
        device: str = typer.Option("auto", "--device", help="Default device override (auto|cpu|cuda)"),
        gen_device: str = typer.Option("", "--gen-device", help="Generation device override"),
        train_device: str = typer.Option("", "--train-device", help="Training device override"),
        teacher_device: str = typer.Option("", "--teacher-device", help="Teacher device override for generation tests"),
        gen_target_util: int = typer.Option(90, "--gen-target-util", help="Target GPU utilization for generation (%)"),
        train_target_util: int = typer.Option(90, "--train-target-util", help="Target GPU utilization for training (%)"),
        util_tolerance: int = typer.Option(5, "--util-tolerance", help="Allowed deviation from target utilization (%)"),
        min_batch_size: int = typer.Option(1, "--min-batch", help="Minimum starting batch size"),
        max_batch_size: int = typer.Option(64, "--max-batch", help="Maximum batch size to consider"),
        growth_factor: float = typer.Option(2.0, "--growth-factor", help="Batch size growth multiplier"),
        monitor_interval: float = typer.Option(1.0, "--monitor-interval", help="GPU monitor sampling interval (seconds)"),
        strict: bool = typer.Option(False, "--strict/--no-strict", help="Enforce strict device handling (no fallbacks)"),
        no_multi_gpu: bool = typer.Option(False, "--no-multi-gpu", help="Disable multi-GPU optimization"),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimize output"),
        config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Load configuration from JSON file")
    ):
        """
        Run optimization to find optimal batch sizes for generation and training.
        
        This command tests different batch sizes to find the optimal configuration
        for both generation and training workloads. It works independently of the GUI
        and provides the same optimization capabilities through the command line.
        
        Examples:
        
        Basic optimization:
        aios optimize --model artifacts/hf_implant/base_model
        
        Custom batch sizes and duration:
        aios optimize --model artifacts/hf_implant/base_model --batch-sizes "1,2,4,8,16" --test-duration 15
        
        Multi-GPU optimization:
        aios optimize --model artifacts/hf_implant/base_model --cuda-devices "0,1"
        
        Load from config file:
        aios optimize --config optimization_config.json
        """
        
        # Load from config file if provided
        if config_file:
            config_path = Path(config_file)
            if not config_path.exists():
                typer.echo(f"❌ Config file not found: {config_file}", err=True)
                raise typer.Exit(1)
                
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Create config from file
                config = OptimizationConfig(**config_data)
                typer.echo(f"📄 Loaded configuration from {config_file}")
                
            except Exception as e:
                typer.echo(f"❌ Failed to load config file: {e}", err=True)
                raise typer.Exit(1)
        else:
            # Create config from command line arguments
            batch_list = [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]
            
            config = OptimizationConfig(
                model=model,
                teacher_model=teacher or model,
                max_seq_len=max_seq,
                test_duration=test_duration,
                max_timeout=max_timeout,
                batch_sizes=batch_list,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                batch_growth_factor=growth_factor,
                output_dir=output_dir,
                cuda_devices=cuda_devices,
                device=device,
                target_util=train_target_util,
                util_tolerance=util_tolerance,
                monitor_interval=monitor_interval,
                strict=strict,
                use_multi_gpu=not no_multi_gpu,
                log_callback=None if quiet else lambda msg: typer.echo(msg)
            )
        
        try:
            # Run optimization
            typer.echo("🚀 Starting optimization process...")
            typer.echo(f"Model: {config.model}")
            typer.echo(f"Batch sizes: {config.batch_sizes}")
            typer.echo(f"Test duration: {config.test_duration}s per batch size")
            typer.echo(f"Output directory: {config.output_dir}")
            
            results, optimizer = optimize_from_config(config)
            
            # Display results
            typer.echo("\n🎉 Optimization Complete!")
            typer.echo("=" * 40)
            
            train_result = results.get("training", {})
            
            if train_result.get("success"):
                typer.echo(f"✅ Training optimal batch size: {train_result.get('optimal_batch', 'N/A')}")
                typer.echo(f"   Max throughput: {train_result.get('max_throughput', 0):.2f} steps/sec")
            else:
                typer.echo("❌ Training optimization failed")
            
            # Show results file location
            results_file = Path(config.output_dir) / f"results_{results.get('session_id', 'unknown')}.json"
            typer.echo(f"\n📁 Detailed results saved to: {results_file}")
            
            # Display any errors
            errors = results.get("errors", [])
            if errors:
                typer.echo("\n⚠️  Errors encountered:")
                for error in errors:
                    typer.echo(f"   • {error}")
            
        except KeyboardInterrupt:
            typer.echo("\n\n⚠️  Optimization interrupted by user")
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"\n❌ Optimization failed: {e}", err=True)
            raise typer.Exit(1)
    
    @app.command("optimize-config")
    def create_config_command(
        output: str = typer.Option("optimization_config.json", "--output", "-o", help="Output configuration file"),
        model: str = typer.Option("base_model", "--model", help="Default model"),
        batch_sizes: str = typer.Option("1,2,4,8,16", "--batch-sizes", help="Default batch sizes"),
        test_duration: int = typer.Option(15, "--test-duration", help="Default test duration")
    ):
        """
        Create a template optimization configuration file.
        
        This creates a JSON configuration file that can be used with the
        'aios optimize --config' command for repeatable optimizations.
        """
        
        batch_list = [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]
        
        config = OptimizationConfig(
            model=model,
            teacher_model="",
            max_seq_len=512,
            test_duration=test_duration,
            max_timeout=test_duration + 10,
            batch_sizes=batch_list,
            use_multi_gpu=True,
            cuda_devices="",
            output_dir="artifacts/optimization"
        )
        
        # Convert to dict and save
        config_dict = {
            k: v for k, v in config.__dict__.items() 
            if k not in ['log_callback', 'stop_callback']  # Skip non-serializable callbacks
        }
        
        output_path = Path(output)
        try:
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            typer.echo(f"📄 Configuration template created: {output_path}")
            typer.echo("\nYou can now edit this file and use it with:")
            typer.echo(f"  aios optimize --config {output}")
            
        except Exception as e:
            typer.echo(f"❌ Failed to create config file: {e}", err=True)
            raise typer.Exit(1)