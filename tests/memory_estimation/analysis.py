"""Analysis and reporting tools for memory estimation test results.

This module provides tools to:
1. Analyze test results and calculate accuracy metrics
2. Identify patterns and problem areas
3. Generate recommendations for improving estimations
4. Create visualizations and reports
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_harness import TestResult, TestConfiguration


@dataclass
class AccuracyReport:
    """Report on estimation accuracy."""
    total_tests: int
    successful_tests: int
    failed_tests: int
    
    # Overall metrics
    mean_vram_accuracy: float
    mean_ram_accuracy: float
    std_vram_accuracy: float
    std_ram_accuracy: float
    
    # Error metrics
    mean_vram_error_gb: float
    mean_ram_error_gb: float
    median_vram_error_gb: float
    median_ram_error_gb: float
    
    # Distribution
    vram_underestimate_count: int
    vram_overestimate_count: int
    ram_underestimate_count: int
    ram_overestimate_count: int
    
    # Thresholds
    tests_above_95_accuracy: int
    tests_above_90_accuracy: int
    tests_below_80_accuracy: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": {
                "total_tests": self.total_tests,
                "successful_tests": self.successful_tests,
                "failed_tests": self.failed_tests,
            },
            "accuracy": {
                "vram": {
                    "mean": round(self.mean_vram_accuracy, 2),
                    "std": round(self.std_vram_accuracy, 2),
                },
                "ram": {
                    "mean": round(self.mean_ram_accuracy, 2),
                    "std": round(self.std_ram_accuracy, 2),
                },
            },
            "errors": {
                "vram_gb": {
                    "mean": round(self.mean_vram_error_gb, 3),
                    "median": round(self.median_vram_error_gb, 3),
                },
                "ram_gb": {
                    "mean": round(self.mean_ram_error_gb, 3),
                    "median": round(self.median_ram_error_gb, 3),
                },
            },
            "distribution": {
                "vram_underestimate": self.vram_underestimate_count,
                "vram_overestimate": self.vram_overestimate_count,
                "ram_underestimate": self.ram_underestimate_count,
                "ram_overestimate": self.ram_overestimate_count,
            },
            "thresholds": {
                "above_95_pct": self.tests_above_95_accuracy,
                "above_90_pct": self.tests_above_90_accuracy,
                "below_80_pct": self.tests_below_80_accuracy,
            },
        }


class ResultAnalyzer:
    """Analyzes test results to identify patterns and issues."""
    
    def __init__(self, results: List[TestResult]):
        """Initialize analyzer with test results.
        
        Args:
            results: List of TestResult objects
        """
        self.results = results
        self.successful_results = [r for r in results if r.success]
    
    def generate_accuracy_report(self) -> AccuracyReport:
        """Generate overall accuracy report."""
        if not self.successful_results:
            raise ValueError("No successful test results to analyze")
        
        # Calculate statistics
        vram_accuracies = [r.vram_accuracy_pct for r in self.successful_results]
        ram_accuracies = [r.ram_accuracy_pct for r in self.successful_results]
        vram_errors = [r.vram_error_gb for r in self.successful_results]
        ram_errors = [r.ram_error_gb for r in self.successful_results]
        
        # Overall accuracy (use minimum of VRAM and RAM)
        overall_accuracies = [min(r.vram_accuracy_pct, r.ram_accuracy_pct) for r in self.successful_results]
        
        return AccuracyReport(
            total_tests=len(self.results),
            successful_tests=len(self.successful_results),
            failed_tests=len(self.results) - len(self.successful_results),
            mean_vram_accuracy=self._mean(vram_accuracies),
            mean_ram_accuracy=self._mean(ram_accuracies),
            std_vram_accuracy=self._std(vram_accuracies),
            std_ram_accuracy=self._std(ram_accuracies),
            mean_vram_error_gb=self._mean(vram_errors),
            mean_ram_error_gb=self._mean(ram_errors),
            median_vram_error_gb=self._median(vram_errors),
            median_ram_error_gb=self._median(ram_errors),
            vram_underestimate_count=sum(1 for e in vram_errors if e < 0),
            vram_overestimate_count=sum(1 for e in vram_errors if e > 0),
            ram_underestimate_count=sum(1 for e in ram_errors if e < 0),
            ram_overestimate_count=sum(1 for e in ram_errors if e > 0),
            tests_above_95_accuracy=sum(1 for a in overall_accuracies if a >= 95),
            tests_above_90_accuracy=sum(1 for a in overall_accuracies if a >= 90),
            tests_below_80_accuracy=sum(1 for a in overall_accuracies if a < 80),
        )
    
    def analyze_by_optimization(self) -> Dict[str, Dict[str, float]]:
        """Analyze accuracy by optimization type.
        
        Returns:
            Dict mapping optimization name to accuracy metrics
        """
        # Group by optimization flags
        groups = {
            "amp": defaultdict(list),
            "gradient_checkpointing": defaultdict(list),
            "chunking": defaultdict(list),
            "lora": defaultdict(list),
            "8bit_optimizer": defaultdict(list),
            "cpu_offload": defaultdict(list),
            "zero_stage": defaultdict(list),
        }
        
        for result in self.successful_results:
            config = result.config
            
            # Group by each optimization
            groups["amp"][config.use_amp].append(result)
            groups["gradient_checkpointing"][config.use_gradient_checkpointing].append(result)
            groups["chunking"][config.use_chunking].append(result)
            groups["lora"][config.use_lora].append(result)
            groups["8bit_optimizer"][config.use_8bit_optimizer].append(result)
            groups["cpu_offload"][config.offload_optimizer].append(result)
            groups["zero_stage"][config.zero_stage].append(result)
        
        # Calculate stats for each group
        analysis = {}
        
        for opt_name, opt_groups in groups.items():
            analysis[opt_name] = {}
            
            for value, results in opt_groups.items():
                if not results:
                    continue
                
                vram_accuracies = [r.vram_accuracy_pct for r in results]
                ram_accuracies = [r.ram_accuracy_pct for r in results]
                vram_errors = [r.vram_error_gb for r in results]
                
                key = str(value) if not isinstance(value, bool) else ("enabled" if value else "disabled")
                
                analysis[opt_name][key] = {
                    "count": len(results),
                    "mean_vram_accuracy": round(self._mean(vram_accuracies), 2),
                    "mean_ram_accuracy": round(self._mean(ram_accuracies), 2),
                    "mean_vram_error_gb": round(self._mean(vram_errors), 3),
                    "std_vram_accuracy": round(self._std(vram_accuracies), 2),
                }
        
        return analysis
    
    def analyze_by_context_size(self) -> Dict[int, Dict[str, float]]:
        """Analyze accuracy by context size.
        
        Returns:
            Dict mapping context size to accuracy metrics
        """
        groups = defaultdict(list)
        
        for result in self.successful_results:
            groups[result.config.seq_len].append(result)
        
        analysis = {}
        
        for seq_len, results in sorted(groups.items()):
            vram_accuracies = [r.vram_accuracy_pct for r in results]
            ram_accuracies = [r.ram_accuracy_pct for r in results]
            vram_errors = [r.vram_error_gb for r in results]
            
            analysis[seq_len] = {
                "count": len(results),
                "mean_vram_accuracy": round(self._mean(vram_accuracies), 2),
                "mean_ram_accuracy": round(self._mean(ram_accuracies), 2),
                "mean_vram_error_gb": round(self._mean(vram_errors), 3),
                "std_vram_accuracy": round(self._std(vram_accuracies), 2),
            }
        
        return analysis
    
    def analyze_by_tokenizer(self) -> Dict[str, Dict[str, float]]:
        """Analyze accuracy by tokenizer (vocab size impact).
        
        Returns:
            Dict mapping tokenizer name to accuracy metrics
        """
        groups = defaultdict(list)
        
        for result in self.successful_results:
            groups[result.config.tokenizer_name].append(result)
        
        analysis = {}
        
        for tokenizer, results in sorted(groups.items()):
            vram_accuracies = [r.vram_accuracy_pct for r in results]
            ram_accuracies = [r.ram_accuracy_pct for r in results]
            vram_errors = [r.vram_error_gb for r in results]
            
            # Get vocab size from first result
            vocab_size = results[0].config.vocab_size if results else 0
            
            analysis[tokenizer] = {
                "count": len(results),
                "vocab_size": vocab_size,
                "mean_vram_accuracy": round(self._mean(vram_accuracies), 2),
                "mean_ram_accuracy": round(self._mean(ram_accuracies), 2),
                "mean_vram_error_gb": round(self._mean(vram_errors), 3),
                "std_vram_accuracy": round(self._std(vram_accuracies), 2),
            }
        
        return analysis
    
    def identify_problem_cases(self, accuracy_threshold: float = 80.0) -> List[TestResult]:
        """Identify test cases with low accuracy.
        
        Args:
            accuracy_threshold: Threshold below which cases are considered problems
        
        Returns:
            List of TestResult objects with accuracy below threshold
        """
        problem_cases = []
        
        for result in self.successful_results:
            # Consider both VRAM and RAM accuracy
            min_accuracy = min(result.vram_accuracy_pct, result.ram_accuracy_pct)
            
            if min_accuracy < accuracy_threshold:
                problem_cases.append(result)
        
        # Sort by worst accuracy first
        problem_cases.sort(key=lambda r: min(r.vram_accuracy_pct, r.ram_accuracy_pct))
        
        return problem_cases
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving estimation accuracy.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.successful_results:
            return ["No successful results to analyze"]
        
        # Analyze patterns
        by_optimization = self.analyze_by_optimization()
        by_context = self.analyze_by_context_size()
        by_tokenizer = self.analyze_by_tokenizer()
        problem_cases = self.identify_problem_cases(accuracy_threshold=85.0)
        
        # Check for systematic bias
        vram_errors = [r.vram_error_gb for r in self.successful_results]
        mean_error = self._mean(vram_errors)
        
        if mean_error > 0.5:
            recommendations.append(
                f"⚠️ Systematic overestimation detected (mean error: +{mean_error:.2f} GB). "
                "Consider reducing reservation factors or overhead percentages."
            )
        elif mean_error < -0.5:
            recommendations.append(
                f"⚠️ Systematic underestimation detected (mean error: {mean_error:.2f} GB). "
                "This is dangerous! Increase safety margins."
            )
        
        # Check optimization-specific issues
        for opt_name, opt_data in by_optimization.items():
            if len(opt_data) < 2:
                continue
            
            # Compare enabled vs disabled
            if "enabled" in opt_data and "disabled" in opt_data:
                enabled_acc = opt_data["enabled"]["mean_vram_accuracy"]
                disabled_acc = opt_data["disabled"]["mean_vram_accuracy"]
                
                if abs(enabled_acc - disabled_acc) > 10:
                    worse = "enabled" if enabled_acc < disabled_acc else "disabled"
                    recommendations.append(
                        f"📊 {opt_name} shows significant accuracy difference "
                        f"({abs(enabled_acc - disabled_acc):.1f}% gap). "
                        f"Accuracy is worse when {worse}. Review estimation logic."
                    )
        
        # Check context size scaling
        if len(by_context) >= 3:
            context_sizes = sorted(by_context.keys())
            accuracies = [by_context[cs]["mean_vram_accuracy"] for cs in context_sizes]
            
            # Check if accuracy degrades with context size
            if accuracies[0] - accuracies[-1] > 15:
                recommendations.append(
                    f"📈 Accuracy degrades significantly with context size "
                    f"({accuracies[0]:.1f}% at {context_sizes[0]} -> "
                    f"{accuracies[-1]:.1f}% at {context_sizes[-1]}). "
                    "Review activation memory scaling for long sequences."
                )
        
        # Check vocab size impact
        if len(by_tokenizer) >= 2:
            tokenizers = sorted(by_tokenizer.items(), key=lambda x: x[1]["vocab_size"])
            small_vocab = tokenizers[0][1]
            large_vocab = tokenizers[-1][1]
            
            acc_diff = abs(small_vocab["mean_vram_accuracy"] - large_vocab["mean_vram_accuracy"])
            
            if acc_diff > 10:
                worse = "large" if large_vocab["mean_vram_accuracy"] < small_vocab["mean_vram_accuracy"] else "small"
                recommendations.append(
                    f"📝 Vocabulary size significantly impacts accuracy ({acc_diff:.1f}% difference). "
                    f"Accuracy is worse for {worse} vocabularies. "
                    "Review output logits memory calculation."
                )
        
        # Problem cases summary
        if problem_cases:
            recommendations.append(
                f"🔍 Found {len(problem_cases)} test cases with <85% accuracy. "
                "Review these cases for specific issues."
            )
            
            # Identify common patterns in problem cases
            problem_configs = [p.config for p in problem_cases]
            
            # Check if chunking is common in problems
            chunking_problems = sum(1 for c in problem_configs if c.use_chunking)
            if chunking_problems / len(problem_cases) > 0.7:
                recommendations.append(
                    "⚠️ Chunking appears in 70%+ of problem cases. "
                    "Review chunked training memory estimation, especially chunk overlap."
                )
            
            # Check if long context is common
            long_context_problems = sum(1 for c in problem_configs if c.seq_len > 4096)
            if long_context_problems / len(problem_cases) > 0.7:
                recommendations.append(
                    "⚠️ Long context (>4096) appears in 70%+ of problem cases. "
                    "Review memory scaling for very long sequences."
                )
        
        if not recommendations:
            recommendations.append("✅ No major issues detected. Estimation quality looks good!")
        
        return recommendations
    
    @staticmethod
    def _mean(values: List[float]) -> float:
        """Calculate mean."""
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def _median(values: List[float]) -> float:
        """Calculate median."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    @staticmethod
    def _std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


class ReportGenerator:
    """Generates human-readable reports from analysis."""
    
    def __init__(self, analyzer: ResultAnalyzer):
        """Initialize report generator.
        
        Args:
            analyzer: ResultAnalyzer instance
        """
        self.analyzer = analyzer
    
    def generate_markdown_report(self, output_file: Optional[Path] = None) -> str:
        """Generate comprehensive markdown report.
        
        Args:
            output_file: Optional file path to save report
        
        Returns:
            Markdown report string
        """
        report_lines = [
            "# Memory Estimation Accuracy Report",
            "",
            f"*Generated: {self._timestamp()}*",
            "",
            "## Summary",
            "",
        ]
        
        # Overall accuracy
        accuracy_report = self.analyzer.generate_accuracy_report()
        report_data = accuracy_report.to_dict()
        
        report_lines.extend([
            f"- **Total Tests**: {report_data['summary']['total_tests']}",
            f"- **Successful**: {report_data['summary']['successful_tests']}",
            f"- **Failed**: {report_data['summary']['failed_tests']}",
            "",
            "### Accuracy Metrics",
            "",
            f"- **VRAM Accuracy**: {report_data['accuracy']['vram']['mean']:.1f}% ± {report_data['accuracy']['vram']['std']:.1f}%",
            f"- **RAM Accuracy**: {report_data['accuracy']['ram']['mean']:.1f}% ± {report_data['accuracy']['ram']['std']:.1f}%",
            "",
            "### Error Distribution",
            "",
            f"- **VRAM Errors**: Mean = {report_data['errors']['vram_gb']['mean']:+.2f} GB, Median = {report_data['errors']['vram_gb']['median']:+.2f} GB",
            f"- **RAM Errors**: Mean = {report_data['errors']['ram_gb']['mean']:+.2f} GB, Median = {report_data['errors']['ram_gb']['median']:+.2f} GB",
            "",
            "### Quality Thresholds",
            "",
            f"- **≥95% Accuracy**: {report_data['thresholds']['above_95_pct']} tests ({report_data['thresholds']['above_95_pct']/report_data['summary']['successful_tests']*100:.1f}%)",
            f"- **≥90% Accuracy**: {report_data['thresholds']['above_90_pct']} tests ({report_data['thresholds']['above_90_pct']/report_data['summary']['successful_tests']*100:.1f}%)",
            f"- **<80% Accuracy**: {report_data['thresholds']['below_80_pct']} tests ({report_data['thresholds']['below_80_pct']/report_data['summary']['successful_tests']*100:.1f}%)",
            "",
        ])
        
        # Analysis by optimization
        report_lines.extend([
            "## Analysis by Optimization",
            "",
        ])
        
        by_opt = self.analyzer.analyze_by_optimization()
        for opt_name, opt_data in by_opt.items():
            report_lines.append(f"### {opt_name.replace('_', ' ').title()}")
            report_lines.append("")
            
            for value, metrics in opt_data.items():
                report_lines.extend([
                    f"**{value}**:",
                    f"- Count: {metrics['count']}",
                    f"- VRAM Accuracy: {metrics['mean_vram_accuracy']:.1f}% ± {metrics['std_vram_accuracy']:.1f}%",
                    f"- Mean Error: {metrics['mean_vram_error_gb']:+.2f} GB",
                    "",
                ])
        
        # Analysis by context size
        report_lines.extend([
            "## Analysis by Context Size",
            "",
        ])
        
        by_context = self.analyzer.analyze_by_context_size()
        for seq_len, metrics in sorted(by_context.items()):
            report_lines.extend([
                f"### Context: {seq_len}",
                f"- Count: {metrics['count']}",
                f"- VRAM Accuracy: {metrics['mean_vram_accuracy']:.1f}% ± {metrics['std_vram_accuracy']:.1f}%",
                f"- Mean Error: {metrics['mean_vram_error_gb']:+.2f} GB",
                "",
            ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
        ])
        
        recommendations = self.analyzer.generate_recommendations()
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        report_lines.append("")
        
        # Problem cases
        problem_cases = self.analyzer.identify_problem_cases(accuracy_threshold=85.0)
        if problem_cases:
            report_lines.extend([
                "## Problem Cases (<85% Accuracy)",
                "",
            ])
            
            for i, result in enumerate(problem_cases[:10], 1):  # Show top 10
                min_acc = min(result.vram_accuracy_pct, result.ram_accuracy_pct)
                report_lines.extend([
                    f"### {i}. {result.config.test_name}",
                    f"- Accuracy: {min_acc:.1f}% (VRAM: {result.vram_accuracy_pct:.1f}%, RAM: {result.ram_accuracy_pct:.1f}%)",
                    f"- VRAM Error: {result.vram_error_gb:+.2f} GB",
                    f"- Config: {result.config.seq_len} context, batch={result.config.batch_size}",
                    f"- Optimizations: AMP={result.config.use_amp}, GradCkpt={result.config.use_gradient_checkpointing}, Chunking={result.config.use_chunking}",
                    "",
                ])
        
        report_text = "\n".join(report_lines)
        
        # Save if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_text, encoding="utf-8")
            print(f"✅ Report saved to: {output_file}")
        
        return report_text
    
    @staticmethod
    def _timestamp() -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    # Demo: Analyze results
    from test_harness import MemoryTestHarness
    
    harness = MemoryTestHarness()
    results = harness.load_results()
    
    if not results:
        print("No test results found. Run tests first!")
    else:
        analyzer = ResultAnalyzer(results)
        
        # Generate accuracy report
        accuracy_report = analyzer.generate_accuracy_report()
        print("="*80)
        print("ACCURACY REPORT")
        print("="*80)
        print(json.dumps(accuracy_report.to_dict(), indent=2))
        
        # Generate markdown report
        generator = ReportGenerator(analyzer)
        report_text = generator.generate_markdown_report(
            output_file=Path("artifacts/memory_tests/accuracy_report.md")
        )
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        recommendations = analyzer.generate_recommendations()
        for rec in recommendations:
            print(f"  {rec}")
