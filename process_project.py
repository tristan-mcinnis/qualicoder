"""
Script to process a specific project directory and output results 
in an organized directory structure.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import QualitativeCoder, logger

def process_project(project_name: str):
    """
    Process a project directory and save results in organized structure.
    
    Args:
        project_name: Name of the project directory in inputs/
    """
    try:
        logger.success(f"Starting analysis for project: {project_name}")
        
        # Initialize the qualitative coder with project name
        coder = QualitativeCoder(use_embeddings=True, project_name=project_name)
        
        # Process the project directory
        logger.processing(f"Processing project directory: {project_name}")
        results = coder.process_project_directory(project_name, use_embeddings=True)
        
        # Display brief summary
        logger.success("Analysis completed! Summary:")
        print(f"\n📁 Project: {project_name}")
        print(f"📄 Files processed: {len(results['source_files'])}")
        print(f"🏷️ Total clusters: {len(results['codes'])}")
        
        # Show files processed
        print(f"\n📋 Source files:")
        for i, filename in enumerate(results['source_files'], 1):
            print(f"  {i}. {filename}")
        
        # Save results in all formats within project directory
        logger.processing("Saving results in organized directory structure...")
        saved_files = coder.save_results(results, format='all')
        
        logger.success(f"Results saved in project-specific directory:")
        for format_name, file_path in saved_files.items():
            print(f"  📄 {format_name.upper()}: {file_path}")
        
        # Show directory structure
        print(f"\n📂 Output structure:")
        print(f"  outputs/")
        print(f"  └── {project_name}/")
        for format_name in saved_files.keys():
            ext = {'json': 'json', 'markdown': 'md', 'text': 'txt', 'csv': 'csv'}[format_name]
            print(f"      ├── qualitative_analysis_*.{ext}")
        
        logger.success(f"Project {project_name} processed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Project directory not found: {e}")
        print(f"\nAvailable projects in inputs/:")
        inputs_dir = "./inputs"
        if os.path.exists(inputs_dir):
            for item in os.listdir(inputs_dir):
                if os.path.isdir(os.path.join(inputs_dir, item)):
                    print(f"  - {item}")
        return 1
        
    except Exception as e:
        logger.error(f"Error processing project: {str(e)}")
        return 1
    
    return 0

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python process_project.py <project_name>")
        print("\nExample: python process_project.py ib_aus")
        print("\nAvailable projects:")
        
        inputs_dir = "./inputs"
        if os.path.exists(inputs_dir):
            for item in os.listdir(inputs_dir):
                if os.path.isdir(os.path.join(inputs_dir, item)):
                    print(f"  - {item}")
        return 1
    
    project_name = sys.argv[1]
    return process_project(project_name)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)