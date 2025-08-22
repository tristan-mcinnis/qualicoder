"""
Script specifically for analyzing market research transcripts.
Uses enhanced processing for better insights.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import QualitativeCoder, logger

def analyze_market_research(project_name: str = "ib_aus"):
    """
    Analyze market research transcripts with specialized processing.
    
    Args:
        project_name: Name of the project directory in inputs/
    """
    try:
        logger.success(f"Starting market research analysis for: {project_name}")
        
        # Initialize coder with project name
        coder = QualitativeCoder(use_embeddings=False, project_name=project_name)
        
        # Use the specialized market research processor
        logger.processing("Using specialized market research transcript processing...")
        results = coder.process_market_research_transcripts(project_name)
        
        # Display summary of insights
        logger.success("Analysis completed! Key insights:")
        print("\n" + "="*60)
        print("MARKET RESEARCH INSIGHTS")
        print("="*60)
        
        # Show actual specific insights (not generic categories)
        for file_idx, codes in results['codes'].items():
            print(f"\nTranscript {file_idx}:")
            for theme, theme_data in codes.items():
                if isinstance(theme_data, dict) and 'description' in theme_data:
                    print(f"\n📊 {theme}")
                    print(f"   {theme_data['description']}")
                    
                    # Show specific findings with real quotes
                    sub_themes = theme_data.get('sub_themes', [])
                    for sub in sub_themes[:2]:  # Show top 2 sub-themes
                        if sub.get('priority') == 'high':
                            print(f"\n   🔴 HIGH PRIORITY: {sub.get('sub_code')}")
                            if sub.get('description'):
                                print(f"      {sub.get('description')}")
                            if sub.get('example_quote'):
                                print(f"      Quote: \"{sub.get('example_quote')}\"")
        
        # Save all formats
        logger.processing("Saving analysis results...")
        saved_files = coder.save_results(results, format='all')
        
        print("\n" + "="*60)
        print("FILES SAVED:")
        print("="*60)
        for format_name, file_path in saved_files.items():
            print(f"📄 {format_name}: {file_path}")
        
        # Show actionable summary
        if 'summary' in saved_files:
            print("\n" + "="*60)
            print("CHECK THE SUMMARY FILE FOR ACTIONABLE INSIGHTS:")
            print(saved_files['summary'])
            print("="*60)
        
        logger.success("Market research analysis complete!")
        
    except Exception as e:
        logger.error(f"Error in market research analysis: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure transcript files are in inputs/ib_aus/")
        print("2. Check that files are .txt format")
        print("3. Verify OpenAI API key is set in .env")
        return 1
    
    return 0

if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "ib_aus"
    exit_code = analyze_market_research(project)
    sys.exit(exit_code)