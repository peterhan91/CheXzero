#!/usr/bin/env python3
"""
Quick validation script to test for answer leakage in NEJM cases
"""

import os
import re
from typing import Dict, Any

def parse_nejm_case(content_text: str) -> Dict[str, Any]:
    """Parse NEJM case content to extract clinical context and options"""
    case_data = {}
    
    # Extract ID
    id_match = re.search(r'ID:\s*(\S+)', content_text)
    case_data['id'] = id_match.group(1) if id_match else ""
    
    # Extract question (clinical context)
    question_match = re.search(r'Question:\s*(.+?)(?=Choices:|$)', content_text, re.DOTALL)
    if question_match:
        case_data['clinical_context'] = question_match.group(1).strip()
    else:
        case_data['clinical_context'] = ""
    
    # Extract choices
    choices_section = re.search(r'Choices:\s*(.+?)(?=Answer:|$)', content_text, re.DOTALL)
    if choices_section:
        choices_text = choices_section.group(1).strip()
        choices = []
        for line in choices_text.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                choice = re.sub(r'^\d+\.\s*', '', line)
                choices.append(choice)
        case_data['choices'] = choices
    else:
        case_data['choices'] = []
    
    # Extract answer
    answer_match = re.search(r'Answer:\s*(.+?)(?=Reason:|$)', content_text, re.DOTALL)
    if answer_match:
        case_data['answer'] = answer_match.group(1).strip()
    else:
        case_data['answer'] = ""
    
    return case_data

def check_answer_leakage(clinical_context: str, answer: str, choices: list) -> bool:
    """Check if answer appears in clinical context"""
    context_lower = clinical_context.lower()
    answer_lower = answer.lower()
    
    # Check if exact answer appears in context
    if answer_lower in context_lower:
        return True
    
    # Check if any specific choice appears in context
    for choice in choices:
        choice_lower = choice.lower()
        if choice_lower in context_lower and len(choice_lower) > 10:
            return True
    
    return False

def test_nejm_cases():
    """Test NEJM cases for answer leakage"""
    cases_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/multimodal_nejm/cxr_cases/"
    
    case_dirs = [d for d in os.listdir(cases_dir) if d.startswith('IC')]
    case_dirs.sort()
    
    print(f"Testing {len(case_dirs)} NEJM cases for answer leakage...")
    print("=" * 60)
    
    leakage_count = 0
    
    for case_dir in case_dirs[:10]:  # Test first 10 cases
        content_file = os.path.join(cases_dir, case_dir, 'content.txt')
        
        if os.path.exists(content_file):
            with open(content_file, 'r') as f:
                content = f.read()
            
            case_data = parse_nejm_case(content)
            
            print(f"\n📋 Case: {case_dir}")
            print(f"❓ Context: {case_data['clinical_context']}")
            print(f"✅ Answer: {case_data['answer']}")
            print(f"📝 Choices: {case_data['choices']}")
            
            has_leakage = check_answer_leakage(
                case_data['clinical_context'], 
                case_data['answer'], 
                case_data['choices']
            )
            
            if has_leakage:
                leakage_count += 1
                print("🚨 POTENTIAL LEAKAGE DETECTED!")
            else:
                print("✅ No leakage detected")
    
    print("\n" + "=" * 60)
    print(f"📊 Summary: {leakage_count} out of 10 cases had potential leakage")
    print("=" * 60)

if __name__ == "__main__":
    test_nejm_cases() 