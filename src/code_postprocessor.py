from typing import Dict, List
import json
from collections import Counter
from .logger import logger

class CodePostProcessor:
    """Post-process and analyze generated qualitative codes."""
    
    @staticmethod
    def consolidate_codes(all_codes: Dict) -> Dict:
        """
        Consolidate and analyze codes across clusters.
        
        Args:
            all_codes: Dictionary of codes by cluster
            
        Returns:
            Consolidated analysis
        """
        try:
            # Extract all themes and sub-themes
            all_themes = []
            all_sub_themes = []
            priority_counts = Counter()
            
            for cluster_id, codes in all_codes.items():
                for theme, theme_data in codes.items():
                    all_themes.append(theme)
                    
                    # Handle new format with descriptions
                    if isinstance(theme_data, dict) and 'sub_themes' in theme_data:
                        sub_themes = theme_data['sub_themes']
                    else:
                        # Handle old format
                        sub_themes = theme_data if isinstance(theme_data, list) else []
                    
                    for sub_theme in sub_themes:
                        all_sub_themes.append(sub_theme.get('sub_code', 'Unknown'))
                        priority_counts[sub_theme.get('priority', 'unknown')] += 1
            
            # Find common themes across clusters
            theme_counts = Counter(all_themes)
            common_themes = [theme for theme, count in theme_counts.items() if count > 1]
            
            consolidated = {
                'total_clusters': len(all_codes),
                'total_themes': len(set(all_themes)),
                'total_sub_themes': len(all_sub_themes),
                'common_themes': common_themes,
                'priority_distribution': dict(priority_counts),
                'theme_frequency': dict(theme_counts.most_common(10))
            }
            
            logger.success("Successfully consolidated codes")
            return consolidated
            
        except Exception as e:
            logger.error(f"Error consolidating codes: {str(e)}")
            return {}
    
    @staticmethod
    def generate_code_hierarchy(all_codes: Dict) -> Dict:
        """
        Create a hierarchical structure of all codes.
        
        Args:
            all_codes: Dictionary of codes by cluster
            
        Returns:
            Hierarchical code structure
        """
        try:
            hierarchy = {
                'root': 'Qualitative Analysis',
                'clusters': []
            }
            
            for cluster_id, codes in all_codes.items():
                cluster_node = {
                    'cluster_id': cluster_id,
                    'themes': []
                }
                
                for theme, theme_data in codes.items():
                    # Handle new format with descriptions
                    if isinstance(theme_data, dict) and 'sub_themes' in theme_data:
                        sub_themes = theme_data['sub_themes']
                        description = theme_data.get('description', '')
                    else:
                        sub_themes = theme_data if isinstance(theme_data, list) else []
                        description = ''
                    
                    theme_node = {
                        'name': theme,
                        'description': description,
                        'sub_themes': []
                    }
                    
                    for sub_theme in sub_themes:
                        theme_node['sub_themes'].append({
                            'name': sub_theme.get('sub_code', 'Unknown'),
                            'description': sub_theme.get('description', ''),
                            'priority': sub_theme.get('priority', 'unknown')
                        })
                    
                    cluster_node['themes'].append(theme_node)
                
                hierarchy['clusters'].append(cluster_node)
            
            logger.success("Successfully generated code hierarchy")
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error generating hierarchy: {str(e)}")
            return {}
    
    @staticmethod
    def prioritize_findings(all_codes: Dict) -> List[Dict]:
        """
        Extract and prioritize key findings.
        
        Args:
            all_codes: Dictionary of codes by cluster
            
        Returns:
            List of prioritized findings
        """
        try:
            findings = []
            
            for cluster_id, codes in all_codes.items():
                for theme, theme_data in codes.items():
                    # Handle new format with descriptions
                    if isinstance(theme_data, dict) and 'sub_themes' in theme_data:
                        sub_themes = theme_data['sub_themes']
                    else:
                        sub_themes = theme_data if isinstance(theme_data, list) else []
                    
                    for sub_theme in sub_themes:
                        findings.append({
                            'cluster': cluster_id,
                            'theme': theme,
                            'sub_theme': sub_theme.get('sub_code', 'Unknown'),
                            'description': sub_theme.get('description', ''),
                            'priority': sub_theme.get('priority', 'unknown'),
                            'priority_score': {'high': 3, 'medium': 2, 'low': 1}.get(sub_theme.get('priority', 'unknown'), 0)
                        })
            
            # Sort by priority score (high to low)
            findings.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Get top findings
            top_findings = findings[:10] if len(findings) > 10 else findings
            
            logger.success(f"Extracted {len(top_findings)} top findings")
            return top_findings
            
        except Exception as e:
            logger.error(f"Error prioritizing findings: {str(e)}")
            return []
    
    @staticmethod
    def generate_insights(all_codes: Dict, 
                         consolidated: Dict) -> List[str]:
        """
        Generate analytical insights from the codes.
        
        Args:
            all_codes: Dictionary of codes by cluster
            consolidated: Consolidated analysis
            
        Returns:
            List of insights
        """
        try:
            insights = []
            
            # Insight 1: Priority distribution
            if consolidated.get('priority_distribution'):
                high_priority = consolidated['priority_distribution'].get('high', 0)
                total = sum(consolidated['priority_distribution'].values())
                if total > 0:
                    high_pct = (high_priority / total) * 100
                    insights.append(f"High-priority items represent {high_pct:.1f}% of all sub-themes, indicating areas requiring immediate attention")
            
            # Insight 2: Common themes
            if consolidated.get('common_themes'):
                insights.append(f"Found {len(consolidated['common_themes'])} themes that appear across multiple clusters, suggesting cross-cutting issues")
            
            # Insight 3: Theme diversity
            if consolidated.get('total_themes') and consolidated.get('total_clusters'):
                avg_themes = consolidated['total_themes'] / consolidated['total_clusters']
                insights.append(f"Average of {avg_themes:.1f} unique themes per cluster indicates {'high' if avg_themes > 3 else 'moderate'} thematic diversity")
            
            # Insight 4: Most frequent theme
            if consolidated.get('theme_frequency'):
                top_theme = list(consolidated['theme_frequency'].keys())[0]
                insights.append(f"'{top_theme}' emerged as the most prominent theme across the analysis")
            
            logger.success(f"Generated {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    @staticmethod
    def export_for_visualization(hierarchy: Dict) -> str:
        """
        Export hierarchy in a format suitable for visualization tools.
        
        Args:
            hierarchy: Hierarchical code structure
            
        Returns:
            JSON string for visualization
        """
        try:
            # Format for D3.js or similar visualization libraries
            viz_data = {
                'name': hierarchy.get('root', 'Analysis'),
                'children': []
            }
            
            for cluster in hierarchy.get('clusters', []):
                cluster_node = {
                    'name': f"Cluster {cluster['cluster_id']}",
                    'children': []
                }
                
                for theme in cluster.get('themes', []):
                    theme_node = {
                        'name': theme['name'],
                        'children': []
                    }
                    
                    for sub_theme in theme.get('sub_themes', []):
                        theme_node['children'].append({
                            'name': sub_theme['name'],
                            'value': {'high': 3, 'medium': 2, 'low': 1}.get(sub_theme['priority'], 1),
                            'priority': sub_theme['priority']
                        })
                    
                    cluster_node['children'].append(theme_node)
                
                viz_data['children'].append(cluster_node)
            
            return json.dumps(viz_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting for visualization: {str(e)}")
            return "{}"