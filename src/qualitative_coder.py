import os
import json
from typing import List, Dict, Optional
from datetime import datetime

from .config import Config
from .logger import logger
from .preprocessor import TextPreprocessor
from .chunker import TextChunker
from .transcript_processor import TranscriptProcessor
from .embeddings import EmbeddingGenerator
from .local_vector_store import LocalVectorStore
from .code_generator import QualitativeCodeGenerator
from .code_postprocessor import CodePostProcessor
from .exporters import ResultExporter

class QualitativeCoder:
    """Main orchestration class for qualitative coding analysis."""
    
    def __init__(self, use_embeddings: bool = True, project_name: str = None):
        """
        Initialize the qualitative coder with all components.
        
        Args:
            use_embeddings: Whether to use embeddings for similarity search
            project_name: Name of the project for organizing outputs
        """
        self.project_name = project_name
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker()
        self.code_generator = QualitativeCodeGenerator()
        self.postprocessor = CodePostProcessor()
        
        # Initialize embedding components if needed
        self.use_embeddings = use_embeddings and Config.HUGGING_FACE_TOKEN
        if self.use_embeddings:
            try:
                self.embedding_generator = EmbeddingGenerator()
                self.vector_store = LocalVectorStore()
            except Exception as e:
                logger.warning(f"Could not initialize embeddings: {str(e)}")
                self.use_embeddings = False
                self.embedding_generator = None
                self.vector_store = None
        else:
            self.embedding_generator = None
            self.vector_store = None
            if use_embeddings and not Config.HUGGING_FACE_TOKEN:
                logger.warning("Embeddings requested but HUGGING_FACE_TOKEN not set")
        
        logger.success("Qualitative coder initialized successfully")
    
    def process_texts(self, 
                     texts: List[str], 
                     languages: List[str] = None,
                     cluster_ids: List[int] = None,
                     store_vectors: bool = True) -> Dict:
        """
        Process texts through the complete qualitative coding pipeline.
        
        Args:
            texts: List of texts to analyze
            languages: List of language codes for each text
            cluster_ids: Optional cluster IDs for each text
            store_vectors: Whether to store vectors in Pinecone
            
        Returns:
            Complete analysis results
        """
        try:
            logger.success("Starting qualitative coding analysis...")
            
            # Set defaults
            if languages is None:
                languages = ['en'] * len(texts)
            if cluster_ids is None:
                cluster_ids = [1] * len(texts)
            
            # Step 1: Preprocess texts
            logger.processing("Step 1: Preprocessing texts...")
            cleaned_texts = self.preprocessor.preprocess_texts(texts, languages)
            
            # Step 2: Chunk texts
            logger.processing("Step 2: Chunking texts...")
            chunked_texts = self.chunker.chunk_texts(cleaned_texts)
            
            # Step 3: Generate embeddings (optional)
            embeddings = []
            if self.use_embeddings:
                logger.processing("Step 3: Generating embeddings...")
                embeddings = self.embedding_generator.generate_embeddings(chunked_texts)
                
                # Store vectors locally
                if store_vectors and self.vector_store:
                    logger.processing("Step 4: Storing vectors locally...")
                    vector_ids = [f"chunk_{i}" for i in range(len(chunked_texts))]
                    metadata = [
                        {
                            'text': text[:200] + '...' if len(text) > 200 else text,
                            'cluster_id': cluster_ids[i % len(cluster_ids)],
                            'length': len(text)
                        } 
                        for i, text in enumerate(chunked_texts)
                    ]
                    
                    self.vector_store.add_vectors(embeddings, vector_ids, metadata)
            
            # Step 5: Group texts by cluster
            logger.processing("Step 5: Grouping texts by cluster...")
            clustered_texts = self._group_texts_by_cluster(chunked_texts, cluster_ids, embeddings)
            
            # Step 6: Generate codes for each cluster
            logger.processing("Step 6: Generating qualitative codes...")
            all_codes = {}
            for cluster_id, cluster_data in clustered_texts.items():
                codes = self.code_generator.generate_codes_for_cluster(
                    cluster_data['texts'], 
                    cluster_id, 
                    cluster_data['embeddings']
                )
                all_codes[cluster_id] = codes
            
            # Step 7: Post-process codes
            logger.processing("Step 7: Post-processing codes...")
            consolidated = self.postprocessor.consolidate_codes(all_codes)
            hierarchy = self.postprocessor.generate_code_hierarchy(all_codes)
            top_findings = self.postprocessor.prioritize_findings(all_codes)
            insights = self.postprocessor.generate_insights(all_codes, consolidated)
            
            # Step 8: Generate summary
            logger.processing("Step 8: Generating analysis summary...")
            summary_report = self.code_generator.generate_summary_report(all_codes)
            
            # Compile results
            results = {
                'original_texts': texts,
                'processed_texts': cleaned_texts,
                'chunks': chunked_texts,
                'codes': all_codes,
                'consolidated_analysis': consolidated,
                'code_hierarchy': hierarchy,
                'top_findings': top_findings,
                'insights': insights,
                'summary_report': summary_report,
                'chunk_info': self.chunker.get_chunk_info(chunked_texts),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Add embedding stats if used
            if self.use_embeddings and self.vector_store:
                results['vector_stats'] = self.vector_store.get_stats()
            
            logger.success("Qualitative coding analysis completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error in qualitative coding analysis: {str(e)}")
            raise
    
    def _group_texts_by_cluster(self, 
                               texts: List[str], 
                               cluster_ids: List[int], 
                               embeddings: List[List[float]]) -> Dict[int, Dict]:
        """
        Group texts by their cluster IDs.
        
        Args:
            texts: List of text chunks
            cluster_ids: List of cluster IDs
            embeddings: List of embeddings
            
        Returns:
            Dictionary mapping cluster IDs to their texts and embeddings
        """
        clustered_data = {}
        
        for i, (text, cluster_id) in enumerate(zip(texts, cluster_ids)):
            if cluster_id not in clustered_data:
                clustered_data[cluster_id] = {
                    'texts': [],
                    'embeddings': []
                }
            
            clustered_data[cluster_id]['texts'].append(text)
            if i < len(embeddings):
                clustered_data[cluster_id]['embeddings'].append(embeddings[i])
        
        return clustered_data
    
    def save_results(self, results: Dict, filename: str = None, format: str = 'json') -> str:
        """
        Save analysis results to a file in specified format.
        
        Args:
            results: Analysis results to save
            filename: Optional filename (defaults to timestamp-based name)
            format: Output format ('json', 'markdown', 'text', 'csv', 'all')
            
        Returns:
            Path to saved file (or dict of paths if format='all')
        """
        try:
            if format == 'all':
                # Export in all formats
                base_name = filename.replace('.json', '') if filename else None
                exported_files = ResultExporter.export_all_formats(results, base_name, self.project_name)
                logger.success(f"Results exported in all formats: {list(exported_files.keys())}")
                return exported_files
            
            elif format == 'markdown':
                return ResultExporter.export_markdown(results, filename, self.project_name)
            
            elif format == 'text':
                return ResultExporter.export_text(results, filename, self.project_name)
            
            elif format == 'csv':
                return ResultExporter.export_csv(results, filename, self.project_name)
            
            else:  # Default to JSON
                if filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"qualitative_analysis_{timestamp}.json"
                
                # Create project-specific output directory
                output_dir = Config.OUTPUTS_DIR
                if self.project_name:
                    output_dir = os.path.join(Config.OUTPUTS_DIR, self.project_name)
                
                filepath = os.path.join(output_dir, filename)
                os.makedirs(output_dir, exist_ok=True)
                
                # Remove embeddings for JSON serialization (they're large)
                results_copy = results.copy()
                if 'embeddings' in results and results['embeddings']:
                    results_copy['embeddings'] = f"[{len(results['embeddings'])} embeddings removed for file size]"
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(results_copy, f, indent=2, ensure_ascii=False)
                
                logger.success(f"Results saved to: {filepath}")
                return filepath
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def load_texts_from_file(self, filepath: str) -> List[str]:
        """
        Load texts from a file in the inputs directory.
        
        Args:
            filepath: Path to the file (relative to inputs directory)
            
        Returns:
            List of texts
        """
        try:
            full_path = os.path.join(Config.INPUTS_DIR, filepath)
            
            with open(full_path, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'texts' in data:
                        return data['texts']
                    else:
                        return [str(data)]
                else:
                    # Treat as plain text file
                    content = f.read()
                    return [content]
            
        except Exception as e:
            logger.error(f"Error loading texts from file: {str(e)}")
            raise
    
    def search_similar_texts(self, 
                           query_text: str, 
                           top_k: int = 5) -> List[Dict]:
        """
        Search for similar texts in the local vector store.
        
        Args:
            query_text: Text to search for
            top_k: Number of similar texts to return
            
        Returns:
            Search results
        """
        try:
            if not self.use_embeddings or not self.vector_store:
                logger.warning("Embeddings not enabled or no vectors stored")
                return []
            
            logger.processing(f"Searching for similar texts to: '{query_text[:50]}...'")
            
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embeddings([query_text])[0]
            
            # Search in local vector store
            results = self.vector_store.search_similar(query_embedding, top_k)
            
            logger.success(f"Found {len(results)} similar texts")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar texts: {str(e)}")
            return []
    
    def process_project_directory(self, project_dir: str, use_embeddings: bool = True) -> Dict:
        """
        Process all text files in a project directory.
        
        Args:
            project_dir: Directory name within inputs folder (e.g., 'ib_aus')
            use_embeddings: Whether to use embeddings for analysis
            
        Returns:
            Complete analysis results
        """
        try:
            # Set project name for organized outputs
            self.project_name = project_dir
            
            project_path = os.path.join(Config.INPUTS_DIR, project_dir)
            
            if not os.path.exists(project_path):
                raise FileNotFoundError(f"Project directory not found: {project_path}")
            
            logger.processing(f"Processing project directory: {project_dir}")
            
            # Load all text files from the project directory
            texts = []
            file_names = []
            
            for filename in os.listdir(project_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(project_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        texts.append(content)
                        file_names.append(filename)
            
            if not texts:
                raise ValueError(f"No .txt files found in project directory: {project_dir}")
            
            logger.success(f"Loaded {len(texts)} text files from {project_dir}")
            
            # Process the texts
            results = self.process_texts(
                texts=texts,
                languages=['en'] * len(texts),  # Assuming English for now
                cluster_ids=list(range(1, len(texts) + 1)),  # Each file as separate cluster
                store_vectors=use_embeddings
            )
            
            # Add metadata about source files
            results['project_name'] = project_dir
            results['source_files'] = file_names
            results['project_metadata'] = {
                'total_files': len(file_names),
                'project_directory': project_dir,
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing project directory: {str(e)}")
            raise
    
    def process_market_research_transcripts(self, project_dir: str) -> Dict:
        """
        Process market research transcripts with specialized handling.
        
        Args:
            project_dir: Directory name within inputs folder
            
        Returns:
            Market research focused analysis results
        """
        try:
            # Set project name for organized outputs
            self.project_name = project_dir
            
            project_path = os.path.join(Config.INPUTS_DIR, project_dir)
            
            if not os.path.exists(project_path):
                raise FileNotFoundError(f"Project directory not found: {project_path}")
            
            logger.processing(f"Processing market research transcripts: {project_dir}")
            
            # Load project objectives and context if available
            project_context = self._load_project_context(project_path)
            
            # Determine transcript directory
            transcripts_path = os.path.join(project_path, 'transcripts')
            if not os.path.exists(transcripts_path):
                # Fallback to root directory for backward compatibility
                transcripts_path = project_path
                logger.warning("No 'transcripts' subdirectory found, using project root")
            
            # Load all transcript files
            all_codes = {}
            file_names = []
            
            for filename in os.listdir(transcripts_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(transcripts_path, filename)
                    file_names.append(filename)
                    
                    logger.processing(f"Analyzing transcript: {filename}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        transcript = f.read()
                    
                    # Determine participant type from filename
                    participant_type = 'buyer' if 'BUYER' in filename else 'potential' if 'POTENTIAL' in filename else 'unknown'
                    
                    # Use specialized transcript processor
                    processor = TranscriptProcessor()
                    
                    # Extract focused sample for analysis
                    focused_sample = processor.prepare_for_coding(transcript, max_length=12000)
                    
                    # Create larger topic chunks
                    chunks = processor.create_topic_chunks(transcript, chunk_size=8000)
                    
                    # Take the most relevant chunks (limit to avoid API overload)
                    selected_chunks = chunks[:3] if len(chunks) > 3 else chunks
                    
                    # Generate codes for this transcript with context
                    codes = self.code_generator.generate_codes_for_cluster(
                        selected_chunks,
                        cluster_id=file_names.index(filename) + 1,
                        embeddings=None,
                        context=project_context,
                        participant_type=participant_type
                    )
                    
                    all_codes[file_names.index(filename) + 1] = codes
            
            logger.success(f"Analyzed {len(file_names)} transcripts")
            
            # Post-process all codes
            logger.processing("Consolidating insights across transcripts...")
            consolidated = self.postprocessor.consolidate_codes(all_codes)
            hierarchy = self.postprocessor.generate_code_hierarchy(all_codes)
            top_findings = self.postprocessor.prioritize_findings(all_codes)
            insights = self.postprocessor.generate_insights(all_codes, consolidated)
            
            # Generate market research summary
            summary_report = self.code_generator.generate_summary_report(all_codes)
            
            # Compile results
            results = {
                'project_name': project_dir,
                'source_files': file_names,
                'codes': all_codes,
                'consolidated_analysis': consolidated,
                'code_hierarchy': hierarchy,
                'top_findings': top_findings,
                'insights': insights,
                'summary_report': summary_report,
                'analysis_timestamp': datetime.now().isoformat(),
                'project_metadata': {
                    'total_files': len(file_names),
                    'project_directory': project_dir,
                    'analysis_type': 'market_research_transcripts',
                    'processed_timestamp': datetime.now().isoformat()
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing market research transcripts: {str(e)}")
            raise
    
    def _load_project_context(self, project_path: str) -> Dict:
        """
        Load project objectives and context from the objectives directory.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Dictionary containing project context and objectives
        """
        context = {}
        objectives_path = os.path.join(project_path, 'objectives')
        
        if not os.path.exists(objectives_path):
            logger.warning("No objectives directory found, proceeding without context")
            return context
        
        # Load objectives.json if it exists
        objectives_file = os.path.join(objectives_path, 'objectives.json')
        if os.path.exists(objectives_file):
            try:
                with open(objectives_file, 'r', encoding='utf-8') as f:
                    context['objectives'] = json.load(f)
                logger.success("Loaded project objectives from objectives.json")
            except Exception as e:
                logger.warning(f"Could not load objectives.json: {str(e)}")
        
        # Load research_brief.txt if it exists
        brief_file = os.path.join(objectives_path, 'research_brief.txt')
        if os.path.exists(brief_file):
            try:
                with open(brief_file, 'r', encoding='utf-8') as f:
                    context['research_brief'] = f.read()
                logger.success("Loaded research brief")
            except Exception as e:
                logger.warning(f"Could not load research_brief.txt: {str(e)}")
        
        # Load brand_context.json if it exists
        brand_file = os.path.join(objectives_path, 'brand_context.json')
        if os.path.exists(brand_file):
            try:
                with open(brand_file, 'r', encoding='utf-8') as f:
                    context['brand_context'] = json.load(f)
                logger.success("Loaded brand context")
            except Exception as e:
                logger.warning(f"Could not load brand_context.json: {str(e)}")
        
        # Load competitor_analysis.txt if it exists
        competitor_file = os.path.join(objectives_path, 'competitor_analysis.txt')
        if os.path.exists(competitor_file):
            try:
                with open(competitor_file, 'r', encoding='utf-8') as f:
                    context['competitor_analysis'] = f.read()
                logger.success("Loaded competitor analysis")
            except Exception as e:
                logger.warning(f"Could not load competitor_analysis.txt: {str(e)}")
        
        return context