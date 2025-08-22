# QualiCoder - AI-Powered Qualitative Research Analysis

An intelligent qualitative coding system that analyzes market research transcripts, focus groups, and interviews using OpenAI's GPT models to generate actionable insights with research context awareness.

## Features

- **Context-Aware Analysis** - Incorporates research objectives and brand context for targeted insights
- **Hierarchical Coding** - Generates themes with sub-themes, priorities, and speaker-attributed quotes
- **Multi-Format Export** - Outputs in JSON, Markdown, Text, and CSV formats
- **Market Research Focus** - Optimized for competitive analysis, brand perception, and consumer insights
- **Speaker Attribution** - Tracks and attributes quotes to specific participants
- **Project Organization** - Structured input/output management with objectives and transcripts
- **Smart Chunking** - Topic-based text segmentation preserving context
- **Optional Embeddings** - HuggingFace model integration for similarity search

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your API keys
nano .env
```

Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Organize Your Project

```
inputs/
└── your_project/
    ├── objectives/
    │   ├── objectives.json       # Research objectives
    │   ├── brand_context.json    # Brand positioning
    │   └── research_brief.txt    # Business context
    └── transcripts/
        └── transcript1.txt
```

### 4. Run Analysis

```bash
# Analyze market research project
python analyze_market_research.py your_project
```

## Directory Structure

```
quali_codes/
├── src/                    # Source code modules
│   ├── qualitative_coder.py   # Main orchestration
│   ├── code_generator.py      # OpenAI integration
│   ├── preprocessor.py        # Text cleaning
│   ├── chunker.py            # Text segmentation
│   ├── code_postprocessor.py # Analysis post-processing
│   ├── embeddings.py         # HuggingFace embeddings (optional)
│   ├── local_vector_store.py # Local similarity search
│   ├── config.py             # Configuration management
│   └── logger.py             # Colored logging
├── tests/                  # Test files (for your tests)
├── inputs/                 # Input text files
├── outputs/                # Analysis results (JSON files)
├── logs/                   # Log files
├── analyze_market_research.py # Main entry point
├── process_project.py      # Project processor
├── requirements.txt        # Dependencies
├── .env.template          # Environment template
└── README.md              # This file
```

## Usage

### Basic Usage

```python
from src import QualitativeCoder

# Initialize the coder
coder = QualitativeCoder()

# Process texts
results = coder.process_texts(
    texts=["Your text data here", "More text..."],
    languages=['en', 'en'],
    cluster_ids=[1, 1],
    store_vectors=True
)

# Save results
coder.save_results(results, "my_analysis.json")
```

### Loading Texts from Files

```python
# Load from input file
texts = coder.load_texts_from_file("my_data.json")

# Process loaded texts
results = coder.process_texts(texts)
```

### Similarity Search (Optional)

```python
# Search for similar texts (requires embeddings)
similar = coder.search_similar_texts("mental health", top_k=5)
```

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `HUGGING_FACE_TOKEN` | No | For embeddings/similarity search |
| `CHUNK_SIZE` | No | Max characters per chunk (default: 512) |
| `CHUNK_OVERLAP` | No | Character overlap between chunks (default: 128) |
| `OPENAI_MODEL` | No | OpenAI model to use (default: gpt-4o) |
| `EMBEDDING_MODEL` | No | HuggingFace model for embeddings |

### Directory Selection

The system uses these directories by default:
- **Input files**: Place your data in `./inputs/`
- **Output files**: Results saved to `./outputs/`
- **Log files**: Logs written to `./logs/`

To use different directories, modify the paths in your `.env` file or update `Config` class in `src/config.py`.

## Input Formats

### Text Files
Place `.txt` files in `./inputs/` directory:
```
./inputs/interview_data.txt
```

### JSON Files
```json
{
  "texts": [
    "First interview transcript...",
    "Second interview transcript..."
  ]
}
```

Or simple array:
```json
[
  "Text 1",
  "Text 2"
]
```

## Output Format

Results are saved as JSON files in `./outputs/` with structure:

```json
{
  "original_texts": [...],
  "codes": {
    "1": {
      "Theme Name": [
        {"sub_code": "Sub-theme", "priority": "high"}
      ]
    }
  },
  "consolidated_analysis": {...},
  "top_findings": [...],
  "insights": [...],
  "analysis_timestamp": "2025-08-21T18:03:45"
}
```

## Analysis Output

The system generates:

1. **Hierarchical Codes** - Main themes with sub-themes and priorities
2. **Key Insights** - Analytical observations about priority distribution
3. **Top Findings** - Ranked list of high-priority items
4. **Consolidated Analysis** - Cross-cluster theme analysis
5. **Code Hierarchy** - Structured view for visualization tools

## Customization

### Disable Embeddings (Faster)

```python
# Initialize without embeddings for faster processing
coder = QualitativeCoder(use_embeddings=False)
```

### Custom Chunk Settings

Edit `.env` file:
```
CHUNK_SIZE=1024
CHUNK_OVERLAP=256
```

### Different OpenAI Model

```
OPENAI_MODEL=gpt-5-nano
```

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   - Ensure `OPENAI_API_KEY` is set in `.env` file
   - Check your OpenAI account has credits

2. **Embedding Model Errors** (Non-critical)
   - Embeddings are optional - system works without them
   - Add `HUGGING_FACE_TOKEN` to enable similarity search

3. **Permission Errors**
   - Ensure write permissions for `outputs/` and `logs/` directories

### Error Messages

- `Missing required environment variables` - Check your `.env` file
- `Could not initialize embeddings` - Optional feature, system still works
- `Error saving results` - Check directory permissions

## Example Workflow

1. **Prepare Data**: Place interview transcripts in `./inputs/`
2. **Configure**: Set up `.env` with your OpenAI API key
3. **Run Analysis**: `python main.py` or use custom script
4. **Review Results**: Check `./outputs/` for JSON analysis files
5. **Extract Insights**: Use the generated codes and insights for your research

## Dependencies

- `openai>=1.0.0` - OpenAI API integration
- `transformers>=4.21.0` - HuggingFace models (optional)
- `torch>=2.0.0` - Deep learning backend (optional)
- `scikit-learn>=1.3.0` - Local vector operations
- `nltk>=3.8` - Sentence tokenization
- `numpy>=1.24.0` - Numerical operations
- `python-dotenv>=1.0.0` - Environment management
- `termcolor>=2.3.0` - Colored output

## License

This project is for research and educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the log files in `./logs/`
3. Ensure all dependencies are installed correctly
