"""
Main entry point for the qualitative coding system.
Demonstrates usage with sample data.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import QualitativeCoder, logger

def main():
    """Main function demonstrating the qualitative coding system."""
    
    # Sample texts for demonstration
    sample_texts = [
        """
        The mental health of athletes has become increasingly important in professional sports. 
        Many athletes report experiencing anxiety and depression due to the constant pressure to perform.
        The demanding schedule of training sessions and competitions often leaves little time for personal life and recovery.
        Coaches are now implementing mental health programs alongside physical training regimens.
        """,
        """
        Team dynamics play a crucial role in athletic performance. 
        Athletes who feel supported by their teammates show better resilience during challenging times.
        Regular team-building activities and open communication channels have proven effective in building trust.
        However, maintaining team cohesion during losing streaks remains a significant challenge.
        """,
        """
        The integration of technology in sports training has revolutionized how athletes prepare for competition.
        Wearable devices track performance metrics and recovery patterns in real-time.
        Data analytics help in identifying areas for improvement and preventing potential injuries.
        Some athletes feel overwhelmed by the constant monitoring and data-driven approach to training.
        """
    ]
    
    try:
        logger.success("Starting Qualitative Coding System...")
        
        # Initialize the qualitative coder
        coder = QualitativeCoder()
        
        # Process the sample texts
        logger.processing("Processing sample texts...")
        results = coder.process_texts(
            texts=sample_texts,
            languages=['en', 'en', 'en'],
            cluster_ids=[1, 1, 1],  # All texts in same cluster for demo
            store_vectors=True  # Store vectors locally for similarity search
        )
        
        # Display results
        logger.success("Analysis completed! Here are the results:")
        print("\n" + "="*50)
        print("QUALITATIVE CODES GENERATED:")
        print("="*50)
        
        for cluster_id, codes in results['codes'].items():
            print(f"\nCluster {cluster_id}:")
            for main_theme, theme_data in codes.items():
                print(f"  📌 {main_theme}")
                
                # Handle new format with descriptions
                if isinstance(theme_data, dict) and 'description' in theme_data:
                    print(f"     {theme_data['description']}")
                    sub_themes = theme_data.get('sub_themes', [])
                else:
                    # Handle old format
                    sub_themes = theme_data if isinstance(theme_data, list) else []
                
                for sub_theme in sub_themes:
                    priority = sub_theme.get('priority', 'unknown')
                    sub_code = sub_theme.get('sub_code', 'Unknown')
                    description = sub_theme.get('description', '')
                    print(f"    - {sub_code} (Priority: {priority})")
                    if description:
                        print(f"      {description}")
        
        print("\n" + "="*50)
        print("CHUNK INFORMATION:")
        print("="*50)
        chunk_info = results['chunk_info']
        for key, value in chunk_info.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Display insights
        if 'insights' in results:
            print("\n" + "="*50)
            print("KEY INSIGHTS:")
            print("="*50)
            for insight in results['insights']:
                print(f"• {insight}")
        
        # Display top findings
        if 'top_findings' in results:
            print("\n" + "="*50)
            print("TOP PRIORITY FINDINGS:")
            print("="*50)
            for finding in results['top_findings'][:5]:
                print(f"• [{finding['priority'].upper()}] {finding['theme']} → {finding['sub_theme']}")
        
        # Save results in multiple formats
        logger.processing("Saving results in multiple formats...")
        saved_files = coder.save_results(results, format='all')
        
        logger.success("Results saved in multiple formats:")
        for format_name, file_path in saved_files.items():
            print(f"  📄 {format_name.upper()}: {file_path}")
        
        # Demonstrate similarity search (if vectors were stored)
        if coder.use_embeddings and coder.vector_store:
            logger.processing("Demonstrating similarity search...")
            search_results = coder.search_similar_texts(
                "mental health and stress in athletes", 
                top_k=2
            )
            
            if search_results:
                print("\n" + "="*50)
                print("SIMILARITY SEARCH RESULTS:")
                print("="*50)
                for match in search_results:
                    score = match.get('score', 0)
                    metadata = match.get('metadata', {})
                    text_preview = metadata.get('text', 'No text available')
                    print(f"Score: {score:.3f} | Text: {text_preview}")
        
        logger.success("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nIf you're getting API or configuration errors:")
        print("1. Copy .env.template to .env")
        print("2. Fill in your OpenAI API key in the .env file")
        print("3. Optionally add HuggingFace token for embeddings")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)