'''
Custom evaluation utilities for model comparison
'''
import re
from difflib import SequenceMatcher

def extract_fields(text):
    '''Extract PRIORITY, CATEGORY, TEAM from output'''
    fields = {}
    patterns = {
        'priority': r'\*\*PRIORITY[:\*\s]*\*\*\s*([P0-3])',
        'category': r'\*\*CATEGORY[:\*\s]*\*\*\s*(\w+)',
        'team': r'\*\*TEAM[:\*\s]*\*\*\s*(\w+)'
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields[field] = match.group(1).strip().lower()
    return fields

def evaluate_comparison(baseline_outputs, new_outputs):
    '''Compare quality metrics between baseline and new model'''
    # Field accuracy
    correct = 0
    total = 0
    for baseline, new in zip(baseline_outputs, new_outputs):
        base_fields = extract_fields(baseline)
        new_fields = extract_fields(new)
        total += len(base_fields)
        correct += sum(1 for k, v in base_fields.items() 
                      if k in new_fields and new_fields[k] == v)
    
    field_accuracy = (correct / total * 100) if total > 0 else 0
    
    # Format consistency
    required = ['PRIORITY', 'CATEGORY', 'TEAM', 'SUMMARY', 'ACTION']
    format_scores = [sum(1 for f in required if f in out.upper()) / len(required) 
                    for out in new_outputs]
    format_consistency = sum(format_scores) / len(format_scores) * 100
    
    return {
        'field_accuracy': field_accuracy,
        'format_consistency': format_consistency
    }
