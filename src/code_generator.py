import json
from typing import List, Dict
from openai import OpenAI
from .config import Config
from .logger import logger

class QualitativeCodeGenerator:
    """Generate qualitative codes using OpenAI API."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the qualitative code generator.
        
        Args:
            model_name: OpenAI model to use for code generation
        """
        self.model_name = model_name or Config.OPENAI_MODEL
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.success("Successfully initialized OpenAI client")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
    
    def generate_codes_for_cluster(self, 
                                  texts: List[str], 
                                  cluster_id: int,
                                  embeddings: List[List[float]] = None,
                                  context: Dict = None,
                                  participant_type: str = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate hierarchical qualitative codes for a cluster of texts.
        
        Args:
            texts: List of text chunks to analyze
            cluster_id: ID of the cluster being analyzed
            embeddings: Optional embeddings for the texts
            context: Optional project context including objectives
            participant_type: Type of participant (buyer/potential/unknown)
            
        Returns:
            Dictionary of hierarchical codes with sub-codes and priorities
        """
        try:
            if not self.client:
                return self._get_fallback_codes(cluster_id)
            
            logger.processing(f"Generating codes for cluster {cluster_id} with {len(texts)} texts...")
            
            # Prepare text segments for analysis
            segments_text = "\n".join(f"- {text}" for text in texts)
            
            # Create the analysis prompt with context
            prompt = self._create_analysis_prompt(segments_text, cluster_id, context, participant_type)
            
            # Prepare messages for OpenAI API
            messages = [
                {"role": "system", "content": "You are an expert qualitative researcher specializing in market research and thematic analysis."},
                {"role": "user", "content": prompt}
            ]
            
            # Add embeddings information if available
            if embeddings:
                embeddings_info = f"Embeddings are available for {len(embeddings)} text segments for additional context."
                messages.append({"role": "user", "content": embeddings_info})
            
            # Make API call to OpenAI
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the response
            response_content = completion.choices[0].message.content
            logger.processing("Parsing OpenAI response...")
            
            # Try to extract JSON from the response
            codes = self._parse_response(response_content)
            
            logger.success(f"Successfully generated codes for cluster {cluster_id}")
            return codes
            
        except Exception as e:
            logger.error(f"Error generating codes: {str(e)}")
            return self._get_fallback_codes(cluster_id)
    
    def _create_analysis_prompt(self, segments_text: str, cluster_id: int, context: Dict = None, participant_type: str = None) -> str:
        """
        Create the analysis prompt for OpenAI with project context.
        
        Args:
            segments_text: Formatted text segments
            cluster_id: Cluster ID
            context: Project context including objectives
            participant_type: Type of participant
            
        Returns:
            Formatted prompt string
        """
        # Build context section if available
        context_section = ""
        if context and 'objectives' in context:
            objectives = context['objectives']
            
            # Get relevant objectives based on participant type
            if participant_type == 'buyer' and 'Buyers' in objectives:
                buyer_obj = objectives['Buyers']
                context_section += f"""
RESEARCH CONTEXT FOR BUYERS:
Research Objectives:
{chr(10).join('- ' + obj for obj in buyer_obj.get('Research Objectives', []))}

Key Questions to Address:
{chr(10).join('- ' + q for q in buyer_obj.get('Key Research Questions', []))}
"""
            elif participant_type == 'potential' and 'Potential Buyers' in objectives:
                potential_obj = objectives['Potential Buyers']
                context_section += f"""
RESEARCH CONTEXT FOR POTENTIAL BUYERS:
Research Objectives:
{chr(10).join('- ' + obj for obj in potential_obj.get('Research Objectives', []))}

Key Questions to Address:
{chr(10).join('- ' + q for q in potential_obj.get('Key Research Questions', []))}
"""
        
        # Add brand context if available
        if context and 'brand_context' in context:
            brand_info = context['brand_context']
            context_section += f"""
BRAND CONTEXT:
{json.dumps(brand_info, indent=2)}
"""
        
        prompt = f"""
        You are an expert qualitative market researcher analyzing focus group transcripts about Icebreaker (IB), an outdoor apparel brand.
        
        {context_section}
        
        Transcript segments to analyze (Participant Type: {participant_type or 'unknown'}):
        {segments_text}

        Perform market research coding to identify SPECIFIC insights relevant to the research objectives.
        Focus on insights that directly address the research questions and objectives provided.
        
        Identify 3-4 KEY INSIGHTS that are:
        - DIRECTLY RELEVANT to the research objectives above
        - SPECIFIC to what participants said about Icebreaker vs competitors
        - ACTIONABLE for Icebreaker's business strategy
        - Include REAL quotes from the transcript
        
        Structure your answer as JSON:
        {{
          "themes": [
            {{
              "theme_name": "[Specific IB-relevant insight, e.g., 'IB perceived as technical but lacks lifestyle appeal vs Kathmandu']",
              "theme_description": "[What participants specifically said about IB in this context]",
              "sub_themes": [
                {{
                  "sub_code": "[Specific finding about IB]",
                  "description": "[How this impacts IB's market position and what action it suggests]",
                  "priority": "high/medium/low",
                  "example_quote": "[EXACT quote mentioning IB or relevant competitor - copy paste]",
                  "speaker": "[Speaker name/ID from transcript, e.g., 'P1', 'Moderator', 'Participant 3']"
                }}
              ]
            }}
          ]
        }}
        
        CRITICAL: 
        - Focus on insights specific to Icebreaker's competitive position
        - Address the research objectives and questions provided
        - Extract ACTUAL quotes from the text (don't fabricate)
        - Include the SPEAKER ID/NAME for each quote (look for patterns like "P1:", "Participant 1:", "Speaker:", etc.)
        - If no clear speaker label is found, use "Participant" as default
        - Prioritize based on business impact for Icebreaker
        - Compare IB to mentioned competitors (Kathmandu, Macpac, Patagonia, etc.)
        
        Respond only with the JSON object.
        """
        return prompt
    
    def _parse_response(self, response_content: str) -> Dict:
        """
        Parse the OpenAI response to extract structured codes.
        
        Args:
            response_content: Raw response from OpenAI
            
        Returns:
            Parsed codes dictionary with descriptions
        """
        try:
            # Try to find JSON in the response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_content[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Handle new format with themes array
                if 'themes' in parsed_data:
                    codes = {}
                    for theme in parsed_data['themes']:
                        theme_name = theme.get('theme_name', 'Unknown Theme')
                        codes[theme_name] = {
                            'description': theme.get('theme_description', ''),
                            'sub_themes': theme.get('sub_themes', [])
                        }
                    return codes
                
                # Handle old format (backward compatibility)
                elif isinstance(parsed_data, dict):
                    # Convert old format to new format with minimal descriptions
                    codes = {}
                    for main_code, sub_codes in parsed_data.items():
                        if isinstance(sub_codes, list):
                            codes[main_code] = {
                                'description': f'Theme related to {main_code.lower().replace("_", " ")}',
                                'sub_themes': sub_codes
                            }
                    logger.success("Successfully parsed OpenAI response")
                    return codes
            
            raise ValueError("Could not extract valid JSON from response")
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.warning("Using fallback code generation")
            return {}
    
    def _get_fallback_codes(self, cluster_id: int) -> Dict:
        """
        Generate fallback codes when OpenAI API is unavailable.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Fallback codes dictionary with descriptions
        """
        return {
            "Brand Perception": {
                "description": "How participants view and understand the brand identity and values.",
                "sub_themes": [
                    {"sub_code": "Quality of Products", "description": "Perceptions about product quality and reliability.", "priority": "high"},
                    {"sub_code": "Brand Reputation", "description": "Overall brand standing and trustworthiness in the market.", "priority": "medium"}
                ]
            },
            "Product Range": {
                "description": "Feedback about the variety and features of products offered.",
                "sub_themes": [
                    {"sub_code": "Variety of Offerings", "description": "Range and diversity of products available.", "priority": "high"},
                    {"sub_code": "Innovation Features", "description": "New and unique product capabilities.", "priority": "medium"}
                ]
            },
            "Market Positioning": {
                "description": "How the brand competes and positions itself in the marketplace.",
                "sub_themes": [
                    {"sub_code": "Price Competitiveness", "description": "Value proposition compared to competitors.", "priority": "medium"},
                    {"sub_code": "Target Audience", "description": "Alignment with customer needs and demographics.", "priority": "low"}
                ]
            }
        }
    
    def generate_summary_report(self, all_codes: Dict) -> str:
        """
        Generate a summary report of all generated codes.
        
        Args:
            all_codes: Dictionary mapping cluster IDs to their codes
            
        Returns:
            Formatted summary report
        """
        try:
            logger.processing("Generating summary report...")
            
            report_lines = ["# Qualitative Coding Analysis Report\n"]
            
            for cluster_id, codes in all_codes.items():
                report_lines.append(f"## Cluster {cluster_id}\n")
                
                for main_theme, theme_data in codes.items():
                    report_lines.append(f"### {main_theme}")
                    
                    # Handle new format with descriptions
                    if isinstance(theme_data, dict) and 'description' in theme_data:
                        report_lines.append(f"*{theme_data['description']}*\n")
                        sub_themes = theme_data.get('sub_themes', [])
                    else:
                        # Handle old format
                        sub_themes = theme_data if isinstance(theme_data, list) else []
                    
                    for sub_theme in sub_themes:
                        priority = sub_theme.get('priority', 'unknown')
                        sub_code = sub_theme.get('sub_code', 'Unknown')
                        description = sub_theme.get('description', '')
                        
                        report_lines.append(f"- **{sub_code}** (Priority: {priority})")
                        if description:
                            report_lines.append(f"  - {description}")
                    
                    report_lines.append("")
            
            report = "\n".join(report_lines)
            logger.success("Successfully generated summary report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return "Error generating report"