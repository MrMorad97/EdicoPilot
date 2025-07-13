from typing import Dict, Callable

PROMPT_REGISTRY: Dict[str, Callable] = {}

def register_prompt(func: Callable):
    """Decorator to automatically register prompt functions"""
    PROMPT_REGISTRY[func.__name__] = func
    return func

@register_prompt
def consice():
    return """[STRICT DIRECTIVES]
You MUST follow these rules:
1. Respond with EXACTLY ONE sentence
2. Never exceed 15 words
3. Never use bullet points or lists
4. Be completely factual without embellishment

Context: {context}
Question: {question}

Single-sentence answer:"""

@register_prompt
def detailed():
    return """[COMPREHENSIVE RESPONSE DIRECTIVES]
You MUST follow these rules:
1. Provide thorough, multi-paragraph explanation
2. Include examples and analogies
3. Cover all relevant aspects
4. Use markdown formatting with headings

Context: {context}
Question: {question}

Comprehensive answer:"""