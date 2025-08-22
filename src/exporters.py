import json
import csv
import os
from typing import Dict, List
from datetime import datetime
from .config import Config
from .logger import logger

class ResultExporter:
    """Export analysis results in multiple formats."""
    
    @staticmethod
    def export_markdown(results: Dict, filename: str = None, project_name: str = None) -> str:
        """
        Export results as a Markdown report.
        
        Args:
            results: Analysis results dictionary
            filename: Optional filename (defaults to timestamp-based name)
            project_name: Optional project name for organizing outputs
            
        Returns:
            Path to saved Markdown file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qualitative_analysis_{timestamp}.md"
            
            # Create project-specific output directory
            output_dir = Config.OUTPUTS_DIR
            if project_name:
                output_dir = os.path.join(Config.OUTPUTS_DIR, project_name)
            
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate Markdown content
            md_content = ResultExporter._generate_markdown_content(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.success(f"Markdown report saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting Markdown: {str(e)}")
            raise
    
    @staticmethod
    def export_text(results: Dict, filename: str = None, project_name: str = None) -> str:
        """
        Export results as a plain text report.
        
        Args:
            results: Analysis results dictionary
            filename: Optional filename (defaults to timestamp-based name)
            project_name: Optional project name for organizing outputs
            
        Returns:
            Path to saved text file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qualitative_analysis_{timestamp}.txt"
            
            # Create project-specific output directory
            output_dir = Config.OUTPUTS_DIR
            if project_name:
                output_dir = os.path.join(Config.OUTPUTS_DIR, project_name)
            
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate text content
            text_content = ResultExporter._generate_text_content(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            logger.success(f"Text report saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting text: {str(e)}")
            raise
    
    @staticmethod
    def export_csv(results: Dict, filename: str = None, project_name: str = None) -> str:
        """
        Export codes as CSV for spreadsheet analysis.
        
        Args:
            results: Analysis results dictionary
            filename: Optional filename (defaults to timestamp-based name)
            project_name: Optional project name for organizing outputs
            
        Returns:
            Path to saved CSV file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qualitative_codes_{timestamp}.csv"
            
            # Create project-specific output directory
            output_dir = Config.OUTPUTS_DIR
            if project_name:
                output_dir = os.path.join(Config.OUTPUTS_DIR, project_name)
            
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare CSV data
            csv_data = []
            for cluster_id, codes in results.get('codes', {}).items():
                for theme, sub_themes in codes.items():
                    for sub_theme in sub_themes:
                        csv_data.append({
                            'cluster_id': cluster_id,
                            'theme': theme,
                            'sub_theme': sub_theme.get('sub_code', ''),
                            'priority': sub_theme.get('priority', ''),
                            'priority_score': {'high': 3, 'medium': 2, 'low': 1}.get(sub_theme.get('priority', ''), 0)
                        })
            
            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if csv_data:
                    fieldnames = ['cluster_id', 'theme', 'sub_theme', 'priority', 'priority_score']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)
            
            logger.success(f"CSV file saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            raise
    
    @staticmethod
    def _generate_markdown_content(results: Dict) -> str:
        """Generate Markdown content from results."""
        content = []
        
        # Header
        timestamp = results.get('analysis_timestamp', datetime.now().isoformat())
        content.append(f"# Qualitative Coding Analysis Report")
        content.append(f"*Generated on: {timestamp}*\n")
        
        # Project Information
        if results.get('project_name'):
            content.append(f"**Project**: {results['project_name']}")
            if results.get('source_files'):
                content.append(f"**Files Analyzed**: {len(results['source_files'])} files")
            content.append("")
        
        # Executive Summary
        content.append("## Executive Summary\n")
        content.append("### Quick Overview\n")
        
        # Extract key themes for summary
        all_themes = set()
        high_priority_count = 0
        total_codes = 0
        
        for cluster_id, codes in results.get('codes', {}).items():
            for theme, theme_data in codes.items():
                all_themes.add(theme)
                if isinstance(theme_data, dict):
                    sub_themes = theme_data.get('sub_themes', [])
                else:
                    sub_themes = theme_data if isinstance(theme_data, list) else []
                
                for sub in sub_themes:
                    total_codes += 1
                    if sub.get('priority') == 'high':
                        high_priority_count += 1
        
        content.append(f"- **Key Themes Identified**: {', '.join(sorted(all_themes)) if all_themes else 'None'}")
        content.append(f"- **Total Codes Generated**: {total_codes}")
        content.append(f"- **High Priority Findings**: {high_priority_count}")
        content.append(f"- **Files/Clusters Analyzed**: {len(results.get('codes', {}))}")
        
        content.append("\n### Analysis Metrics\n")
        chunk_info = results.get('chunk_info', {})
        content.append(f"- **Total Text Chunks**: {chunk_info.get('total_chunks', 'N/A')}")
        content.append(f"- **Average Chunk Length**: {chunk_info.get('avg_chunk_length', 0):.0f} characters")
        content.append(f"- **Total Characters**: {chunk_info.get('total_characters', 'N/A'):,}" if isinstance(chunk_info.get('total_characters'), int) else f"- **Total Characters**: {chunk_info.get('total_characters', 'N/A')}")
        
        # Key Insights
        if 'insights' in results and results['insights']:
            content.append("\n## Key Insights\n")
            for insight in results['insights']:
                content.append(f"- {insight}")
        
        # Top Priority Findings
        if 'top_findings' in results and results['top_findings']:
            content.append("\n## Top Priority Findings\n")
            content.append("| Priority | Theme | Sub-Theme |")
            content.append("|----------|-------|-----------|")
            for finding in results['top_findings'][:10]:
                priority = finding.get('priority', '').upper()
                theme = finding.get('theme', '')
                sub_theme = finding.get('sub_theme', '')
                content.append(f"| **{priority}** | {theme} | {sub_theme} |")
        
        # Detailed Codes by Cluster
        content.append("\n## Detailed Analysis by Cluster\n")
        for cluster_id, codes in results.get('codes', {}).items():
            content.append(f"### Cluster {cluster_id}\n")
            
            for theme, theme_data in codes.items():
                content.append(f"#### {theme}\n")
                
                # Handle new format with descriptions
                if isinstance(theme_data, dict) and 'description' in theme_data:
                    content.append(f"*{theme_data['description']}*\n")
                    sub_themes = theme_data.get('sub_themes', [])
                else:
                    # Handle old format (backward compatibility)
                    sub_themes = theme_data if isinstance(theme_data, list) else []
                
                for sub_theme in sub_themes:
                    priority = sub_theme.get('priority', 'unknown')
                    sub_code = sub_theme.get('sub_code', 'Unknown')
                    description = sub_theme.get('description', '')
                    quote = sub_theme.get('example_quote', '')
                    speaker = sub_theme.get('speaker', '')
                    
                    priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')
                    content.append(f"\n{priority_emoji} **{sub_code}** *(Priority: {priority})*")
                    if description:
                        content.append(f"   - {description}")
                    if quote:
                        if speaker:
                            content.append(f"   - {speaker}: \"{quote}\"")
                        else:
                            content.append(f"   - Example: \"{quote}\"")
                content.append("")
        
        # Consolidated Analysis
        if 'consolidated_analysis' in results:
            consolidated = results['consolidated_analysis']
            content.append("## Consolidated Analysis\n")
            content.append(f"- **Total Clusters**: {consolidated.get('total_clusters', 'N/A')}")
            content.append(f"- **Total Unique Themes**: {consolidated.get('total_themes', 'N/A')}")
            content.append(f"- **Total Sub-Themes**: {consolidated.get('total_sub_themes', 'N/A')}")
            
            if consolidated.get('common_themes'):
                content.append(f"- **Common Themes Across Clusters**: {', '.join(consolidated['common_themes'])}")
            
            if consolidated.get('priority_distribution'):
                content.append("\n### Priority Distribution")
                for priority, count in consolidated['priority_distribution'].items():
                    content.append(f"- {priority.title()}: {count}")
        
        # Original Summary Report
        if 'summary_report' in results and results['summary_report']:
            content.append("\n## Detailed Summary\n")
            content.append(results['summary_report'])
        
        # Footer
        content.append("\n---")
        content.append("*Report generated by Qualitative Coding System*")
        
        return "\n".join(content)
    
    @staticmethod
    def export_simplified_summary(results: Dict, filename: str = None, project_name: str = None) -> str:
        """
        Export a simplified, actionable summary of the qualitative coding analysis.
        
        Args:
            results: Analysis results dictionary
            filename: Optional filename (defaults to timestamp-based name)
            project_name: Optional project name for organizing outputs
            
        Returns:
            Path to saved summary file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"coding_summary_{timestamp}.md"
            
            # Create project-specific output directory
            output_dir = Config.OUTPUTS_DIR
            if project_name:
                output_dir = os.path.join(Config.OUTPUTS_DIR, project_name)
            
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate simplified summary content
            content = []
            
            # Header
            content.append("# Qualitative Coding Summary\n")
            if project_name:
                content.append(f"**Project**: {project_name}\n")
            content.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
            
            # Key Themes with Descriptions
            content.append("## Key Themes Identified\n")
            themes_seen = set()
            theme_descriptions = {}
            
            for cluster_id, codes in results.get('codes', {}).items():
                for theme, theme_data in codes.items():
                    if theme not in themes_seen:
                        themes_seen.add(theme)
                        
                        if isinstance(theme_data, dict) and 'description' in theme_data:
                            theme_descriptions[theme] = theme_data['description']
            
            for theme in sorted(themes_seen):
                content.append(f"### {theme}")
                if theme in theme_descriptions:
                    content.append(f"{theme_descriptions[theme]}\n")
                else:
                    content.append("")
            
            # High Priority Action Items
            content.append("## High Priority Findings (Action Required)\n")
            high_priority_items = []
            
            for cluster_id, codes in results.get('codes', {}).items():
                for theme, theme_data in codes.items():
                    if isinstance(theme_data, dict) and 'sub_themes' in theme_data:
                        sub_themes = theme_data['sub_themes']
                    else:
                        sub_themes = theme_data if isinstance(theme_data, list) else []
                    
                    for sub_theme in sub_themes:
                        if sub_theme.get('priority') == 'high':
                            high_priority_items.append({
                                'theme': theme,
                                'sub_code': sub_theme.get('sub_code', 'Unknown'),
                                'description': sub_theme.get('description', ''),
                                'quote': sub_theme.get('example_quote', ''),
                                'speaker': sub_theme.get('speaker', '')
                            })
            
            for item in high_priority_items[:10]:  # Top 10 high priority items
                content.append(f"**{item['theme']} - {item['sub_code']}**")
                if item['description']:
                    content.append(f"- {item['description']}")
                if item['quote']:
                    if item['speaker']:
                        content.append(f"- *{item['speaker']}: \"{item['quote']}\"*")
                    else:
                        content.append(f"- *\"{item['quote']}\"*")
                content.append("")
            
            # Summary Statistics
            content.append("## Summary Statistics\n")
            total_codes = sum(len(theme_data.get('sub_themes', []) if isinstance(theme_data, dict) else theme_data) 
                            for codes in results.get('codes', {}).values() 
                            for theme_data in codes.values())
            
            content.append(f"- Total themes: {len(themes_seen)}")
            content.append(f"- Total codes: {total_codes}")
            content.append(f"- High priority items: {len(high_priority_items)}")
            content.append(f"- Files analyzed: {len(results.get('source_files', []))}")
            
            # Recommendations Section
            content.append("\n## Next Steps\n")
            content.append("1. Review high priority findings and develop action plans")
            content.append("2. Share findings with stakeholders for validation")
            content.append("3. Consider follow-up interviews on key themes")
            content.append("4. Track implementation of improvements based on insights")
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(content))
            
            logger.success(f"Simplified summary saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting simplified summary: {str(e)}")
            raise
    
    @staticmethod
    def _generate_text_content(results: Dict) -> str:
        """Generate plain text content from results."""
        content = []
        
        # Header
        timestamp = results.get('analysis_timestamp', datetime.now().isoformat())
        content.append("=" * 60)
        content.append("QUALITATIVE CODING ANALYSIS REPORT")
        content.append("=" * 60)
        content.append(f"Generated on: {timestamp}")
        content.append("")
        
        # Executive Summary
        content.append("EXECUTIVE SUMMARY")
        content.append("-" * 20)
        chunk_info = results.get('chunk_info', {})
        content.append(f"Total Text Chunks Analyzed: {chunk_info.get('total_chunks', 'N/A')}")
        content.append(f"Average Chunk Length: {chunk_info.get('avg_chunk_length', 0):.0f} characters")
        content.append(f"Total Characters Processed: {chunk_info.get('total_characters', 'N/A')}")
        content.append("")
        
        # Key Insights
        if 'insights' in results and results['insights']:
            content.append("KEY INSIGHTS")
            content.append("-" * 15)
            for i, insight in enumerate(results['insights'], 1):
                content.append(f"{i}. {insight}")
            content.append("")
        
        # Top Priority Findings
        if 'top_findings' in results and results['top_findings']:
            content.append("TOP PRIORITY FINDINGS")
            content.append("-" * 25)
            for i, finding in enumerate(results['top_findings'][:10], 1):
                priority = finding.get('priority', '').upper()
                theme = finding.get('theme', '')
                sub_theme = finding.get('sub_theme', '')
                content.append(f"{i:2d}. [{priority:6s}] {theme} -> {sub_theme}")
            content.append("")
        
        # Detailed Codes by Cluster
        content.append("DETAILED ANALYSIS BY CLUSTER")
        content.append("-" * 35)
        for cluster_id, codes in results.get('codes', {}).items():
            content.append(f"\nCLUSTER {cluster_id}")
            content.append("=" * 15)
            
            for theme, theme_data in codes.items():
                content.append(f"\n{theme}:")
                
                # Handle new format with descriptions
                if isinstance(theme_data, dict) and 'sub_themes' in theme_data:
                    if theme_data.get('description'):
                        content.append(f"  {theme_data['description']}")
                    sub_themes = theme_data.get('sub_themes', [])
                else:
                    sub_themes = theme_data if isinstance(theme_data, list) else []
                
                for sub_theme in sub_themes:
                    priority = sub_theme.get('priority', 'unknown')
                    sub_code = sub_theme.get('sub_code', 'Unknown')
                    description = sub_theme.get('description', '')
                    content.append(f"  - {sub_code} (Priority: {priority})")
                    if description:
                        content.append(f"    {description}")
        
        # Consolidated Analysis
        if 'consolidated_analysis' in results:
            consolidated = results['consolidated_analysis']
            content.append("\n\nCONSOLIDATED ANALYSIS")
            content.append("-" * 25)
            content.append(f"Total Clusters: {consolidated.get('total_clusters', 'N/A')}")
            content.append(f"Total Unique Themes: {consolidated.get('total_themes', 'N/A')}")
            content.append(f"Total Sub-Themes: {consolidated.get('total_sub_themes', 'N/A')}")
            
            if consolidated.get('priority_distribution'):
                content.append("\nPriority Distribution:")
                for priority, count in consolidated['priority_distribution'].items():
                    content.append(f"  {priority.title()}: {count}")
        
        # Footer
        content.append("\n" + "=" * 60)
        content.append("Report generated by Qualitative Coding System")
        content.append("=" * 60)
        
        return "\n".join(content)
    
    @staticmethod
    def export_all_formats(results: Dict, base_name: str = None, project_name: str = None) -> Dict[str, str]:
        """
        Export results in all available formats.
        
        Args:
            results: Analysis results dictionary
            base_name: Base filename (without extension)
            project_name: Optional project name for organizing outputs
            
        Returns:
            Dictionary mapping format names to file paths
        """
        try:
            if base_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"qualitative_analysis_{timestamp}"
            
            exported_files = {}
            
            # Create project-specific output directory
            output_dir = Config.OUTPUTS_DIR
            if project_name:
                output_dir = os.path.join(Config.OUTPUTS_DIR, project_name)
            
            # Export JSON
            json_path = os.path.join(output_dir, f"{base_name}.json")
            os.makedirs(output_dir, exist_ok=True)
            
            # Remove embeddings for JSON serialization
            results_copy = results.copy()
            if 'embeddings' in results and results['embeddings']:
                results_copy['embeddings'] = f"[{len(results['embeddings'])} embeddings removed for file size]"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, indent=2, ensure_ascii=False)
            exported_files['json'] = json_path
            
            # Export Markdown
            exported_files['markdown'] = ResultExporter.export_markdown(results, f"{base_name}.md", project_name)
            
            # Export Text
            exported_files['text'] = ResultExporter.export_text(results, f"{base_name}.txt", project_name)
            
            # Export CSV
            exported_files['csv'] = ResultExporter.export_csv(results, f"{base_name}.csv", project_name)
            
            # Export Simplified Summary
            exported_files['summary'] = ResultExporter.export_simplified_summary(results, f"{base_name}_summary.md", project_name)
            
            logger.success(f"Exported results in {len(exported_files)} formats")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting all formats: {str(e)}")
            return {}