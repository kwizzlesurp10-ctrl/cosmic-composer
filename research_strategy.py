"""
Cognitive Study Assistant - Research Strategy Formulation

This module helps users research and learn by formulating focused research strategies
based on their questions. It analyzes key concepts and terms to determine appropriate
search strategies.
"""

import json
import re
from typing import List, Dict, Any


class ResearchStrategyFormulator:
    """
    Formulates research strategies by analyzing user questions and generating
    appropriate search queries.
    """
    
    def __init__(self):
        """Initialize the research strategy formulator."""
        pass
    
    def extract_key_concepts(self, question: str) -> List[str]:
        """
        Extract key concepts and terms from a user question.
        
        Args:
            question: The user's question string
            
        Returns:
            List of key concepts/terms identified in the question
        """
        # Convert to lowercase for processing
        question_lower = question.lower()
        
        # Common technical terms and concepts (with their canonical forms)
        technical_terms_map = {
            'rag': 'RAG',
            'retrieval augmented generation': 'Retrieval Augmented Generation',
            'llm': 'LLM',
            'large language model': 'Large Language Model',
            'vector search': 'Vector Search',
            'embedding': 'Embedding',
            'transformer': 'Transformer',
            'neural network': 'Neural Network',
            'machine learning': 'Machine Learning',
            'deep learning': 'Deep Learning',
            'nlp': 'NLP',
            'natural language processing': 'Natural Language Processing',
            'fine-tuning': 'Fine-tuning',
            'prompt engineering': 'Prompt Engineering',
            'few-shot learning': 'Few-shot Learning'
        }
        
        # Extract quoted terms (often key concepts) - highest priority
        quoted_terms = re.findall(r'"([^"]+)"', question)
        
        # Check for technical terms in the question
        found_terms = []
        for term_key, term_value in technical_terms_map.items():
            if term_key in question_lower:
                found_terms.append(term_value)
        
        # Extract standalone capitalized acronyms (like RAG, LLM, NLP)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', question)
        
        # Extract multi-word capitalized phrases (but filter out common words)
        capitalized_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question)
        
        # Filter out common capitalized words that aren't concepts
        common_words = {'Can', 'You', 'Tell', 'Me', 'More', 'About', 'The', 'And', 'How', 'It', 'To', 'Via'}
        capitalized_phrases = [p for p in capitalized_phrases if p not in common_words]
        
        # Combine all found terms, prioritizing quoted terms and technical terms
        key_concepts = []
        
        # Add quoted terms first (highest priority)
        key_concepts.extend(quoted_terms)
        
        # Add technical terms found
        key_concepts.extend(found_terms)
        
        # Add acronyms
        key_concepts.extend(acronyms)
        
        # Add capitalized phrases (but avoid duplicates)
        for phrase in capitalized_phrases:
            if phrase not in key_concepts:
                key_concepts.append(phrase)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in key_concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen:
                seen.add(concept_lower)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def generate_search_instructions(self, term: str, question: str) -> str:
        """
        Generate search instructions for a given term based on the question context.
        
        Args:
            term: The search term
            question: The original user question
            
        Returns:
            Instruction string for how to use this search term
        """
        question_lower = question.lower()
        term_lower = term.lower()
        
        # Generate contextual instructions based on question type
        if 'how' in question_lower or 'apply' in question_lower or 'use' in question_lower:
            return f"Explain how {term} works and how it can be applied."
        elif 'what' in question_lower or 'concept' in question_lower:
            return f"Describe the concept and utility of {term}."
        elif 'why' in question_lower:
            return f"Explain why {term} is important and its benefits."
        elif 'compare' in question_lower or 'difference' in question_lower:
            return f"Describe {term} and its characteristics for comparison."
        else:
            return f"Describe the concept and utility of {term}."
    
    def formulate_strategy(self, question: str) -> Dict[str, Any]:
        """
        Formulate a research strategy based on a user question.
        
        Args:
            question: The user's research question
            
        Returns:
            Dictionary containing 'reasoning' and 'searches' fields
        """
        # Extract key concepts
        key_concepts = self.extract_key_concepts(question)
        
        # Expand acronyms and add related terms
        expanded_concepts = self._expand_and_add_related_terms(key_concepts, question)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(question, expanded_concepts)
        
        # Generate search queries
        searches = []
        for term in expanded_concepts[:5]:  # Limit to top 5 most relevant terms
            instructions = self.generate_search_instructions(term, question)
            searches.append({
                "term": term,
                "instructions": instructions
            })
        
        # If no key concepts found, create a general search
        if not searches:
            # Extract main topic from question
            main_topic = question.split('?')[0].split('.')[0].strip()
            searches.append({
                "term": main_topic,
                "instructions": f"Describe the concept and utility of {main_topic}."
            })
        
        return {
            "reasoning": reasoning,
            "searches": searches
        }
    
    def _expand_and_add_related_terms(self, key_concepts: List[str], question: str) -> List[str]:
        """
        Expand acronyms to their full forms and add related terms based on context.
        
        Args:
            key_concepts: List of extracted key concepts
            question: The original user question
            
        Returns:
            Expanded list of concepts with related terms
        """
        # Mapping of acronyms to their expanded forms
        acronym_expansions = {
            'RAG': 'Retrieval Augmented Generation',
            'LLM': 'Large Language Model',
            'NLP': 'Natural Language Processing',
            'AI': 'Artificial Intelligence',
            'ML': 'Machine Learning',
            'DL': 'Deep Learning'
        }
        
        # Related terms based on context
        related_terms_map = {
            'RAG': ['Vector Search'],
            'Retrieval Augmented Generation': ['Vector Search'],
            'LLM': ['Prompt Engineering'],
            'Large Language Model': ['Prompt Engineering']
        }
        
        expanded_concepts = []
        question_lower = question.lower()
        
        # Process each concept
        for concept in key_concepts:
            # Add the concept itself
            if concept not in expanded_concepts:
                expanded_concepts.append(concept)
            
            # Expand acronyms
            if concept in acronym_expansions:
                expanded = acronym_expansions[concept]
                if expanded not in expanded_concepts:
                    expanded_concepts.append(expanded)
            
            # Add related terms
            if concept in related_terms_map:
                for related_term in related_terms_map[concept]:
                    if related_term not in expanded_concepts:
                        expanded_concepts.append(related_term)
        
        # Also check if question mentions RAG/retrieval - add vector search if not already present
        if ('rag' in question_lower or 'retrieval' in question_lower) and 'Vector Search' not in expanded_concepts:
            expanded_concepts.append('Vector Search')
        
        return expanded_concepts
    
    def _generate_reasoning(self, question: str, key_concepts: List[str]) -> str:
        """
        Generate reasoning text explaining the search strategy.
        
        Args:
            question: The user's question
            key_concepts: List of extracted key concepts
            
        Returns:
            Reasoning string
        """
        if not key_concepts:
            return f"The user is asking: '{question}'. I should search for documents related to the main topic to provide a comprehensive response."
        
        # Build a natural-sounding reasoning statement
        question_lower = question.lower()
        
        # Determine if we should mention "concept of" or just the term
        if 'concept' in question_lower:
            primary_concept = f"the concept of {key_concepts[0]}"
        else:
            primary_concept = key_concepts[0]
        
        # Check if question mentions specific applications or contexts
        application_context = ""
        if 'generate' in question_lower and 'answer' in question_lower:
            # Check if LLM is mentioned for "via LLM" context
            if any('llm' in c.lower() or 'large language model' in c.lower() for c in key_concepts):
                application_context = " and its application in generating answers to user questions via LLM"
            else:
                application_context = " and its application in generating answers to user questions"
        elif 'apply' in question_lower or 'application' in question_lower:
            application_context = " and its application"
        
        # Build concept list for search terms (prioritize core concepts, limit to 3-4)
        # Filter to most relevant: prefer acronyms and their expansions, plus key related terms
        search_terms = []
        seen_lower = set()
        
        # Add concepts, avoiding duplicates (case-insensitive)
        for concept in key_concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen_lower:
                search_terms.append(concept)
                seen_lower.add(concept_lower)
                if len(search_terms) >= 4:  # Limit to top 4 for reasoning
                    break
        
        # Build concepts string for reasoning
        if len(search_terms) == 1:
            concepts_str = search_terms[0]
        elif len(search_terms) == 2:
            concepts_str = f"{search_terms[0]} and {search_terms[1]}"
        else:
            # Format: "term1, term2, and term3"
            concepts_str = f"{search_terms[0]}, {', '.join(search_terms[1:-1])}, and {search_terms[-1]}"
        
        # Build reasoning statement
        reasoning = f"The user is asking about {primary_concept}{application_context}. I should search for documents related to {concepts_str} to provide a comprehensive response."
        
        return reasoning


def formulate_research_strategy(question: str) -> str:
    """
    Main function to formulate research strategy from user question.
    Returns JSON string (without markdown formatting).
    
    Args:
        question: The user's research question
        
    Returns:
        JSON string containing reasoning and searches
    """
    formulator = ResearchStrategyFormulator()
    strategy = formulator.formulate_strategy(question)
    return json.dumps(strategy, indent=2)


if __name__ == "__main__":
    # Example usage
    example_question = 'Can you tell me more about the concept of "RAG" and how it can be applied to generate answers to user questions via LLM?'
    result = formulate_research_strategy(example_question)
    print(result)
