#!/usr/bin/env python3
"""
MedGemma-4B-IT Evaluation for NEJM Chest X-ray VQA Cases
Using Google's medical multimodal model for comparison
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                # Remove the number prefix
                choice = re.sub(r'^\d+\.\s*', '', line)
                choices.append(choice)
        case_data['choices'] = choices
    else:
        case_data['choices'] = []
    
    # Extract answer (for evaluation)
    answer_match = re.search(r'Answer:\s*(.+?)(?=Reason:|$)', content_text, re.DOTALL)
    if answer_match:
        case_data['answer'] = answer_match.group(1).strip()
    else:
        case_data['answer'] = ""
    
    return case_data


def load_nejm_cases(cases_dir: str) -> List[Dict[str, Any]]:
    """Load all NEJM cases from the directory"""
    cases = []
    
    case_dirs = [d for d in os.listdir(cases_dir) if d.startswith('IC') and os.path.isdir(os.path.join(cases_dir, d))]
    case_dirs.sort()
    
    print(f"📂 Found {len(case_dirs)} NEJM case directories")
    
    for case_dir in tqdm(case_dirs, desc="🔍 Loading cases"):
        case_path = os.path.join(cases_dir, case_dir)
        
        # Handle both content.txt and context.txt naming conventions
        content_file = os.path.join(case_path, 'content.txt')
        context_file = os.path.join(case_path, 'context.txt')
        image_file = os.path.join(case_path, 'image.png')
        
        # Check for either content.txt or context.txt
        if os.path.exists(content_file):
            text_file = content_file
        elif os.path.exists(context_file):
            text_file = context_file
        else:
            text_file = None
        
        if text_file and os.path.exists(image_file):
            with open(text_file, 'r') as f:
                content_text = f.read()
            
            case_data = parse_nejm_case(content_text)
            case_data['case_dir'] = case_dir
            case_data['image_path'] = image_file
            
            # Only include cases with valid questions and choices
            if case_data['clinical_context'] and len(case_data['choices']) > 0:
                cases.append(case_data)
            else:
                print(f"⚠️ Skipping case {case_dir}: missing context or choices")
    
    print(f"✅ Loaded {len(cases)} valid NEJM cases")
    return cases


def load_medgemma_model():
    """Load MedGemma-4B-IT model"""
    print("🤖 Loading MedGemma-4B-IT model...")
    
    model_id = "google/medgemma-4b-it"
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        print("✅ MedGemma-4B-IT model loaded successfully")
        return model, processor
    
    except Exception as e:
        print(f"❌ Error loading MedGemma model: {e}")
        print("💡 Try installing accelerate: pip install accelerate")
        raise


def create_vqa_prompt(clinical_context: str, choices: List[str]) -> str:
    """Create a VQA prompt for MedGemma"""
    prompt = f"""You are an expert radiologist. Based on the chest X-ray image and clinical context, choose the best answer.

Clinical Context: {clinical_context}

Please select the most appropriate answer from the following choices:
"""
    
    for i, choice in enumerate(choices, 1):
        prompt += f"{i}. {choice}\n"
    
    prompt += "\nProvide only the number of your choice (1, 2, 3, 4, or 5) and the corresponding answer text."
    
    return prompt


def extract_choice_from_response(response: str, choices: List[str]) -> Tuple[int, str]:
    """Extract the chosen answer from MedGemma's response"""
    response = response.strip().lower()
    
    # Look for explicit number choice (1, 2, 3, etc.)
    number_match = re.search(r'\b([1-5])\b', response)
    if number_match:
        choice_num = int(number_match.group(1)) - 1  # Convert to 0-based index
        if 0 <= choice_num < len(choices):
            return choice_num, choices[choice_num]
    
    # Look for choice text in response
    for i, choice in enumerate(choices):
        if choice.lower().strip() in response:
            return i, choice
    
    # If no clear match, try partial matches
    for i, choice in enumerate(choices):
        choice_words = choice.lower().split()
        if len(choice_words) > 1:  # Only for multi-word choices
            if any(word in response for word in choice_words if len(word) > 3):
                return i, choice
    
    # Default to first choice if no match found
    print(f"⚠️ Could not parse response: '{response[:100]}...', defaulting to choice 1")
    return 0, choices[0]


def evaluate_medgemma(model, processor, cases: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluate MedGemma on NEJM VQA cases"""
    print(f"🎯 Evaluating MedGemma on {len(cases)} cases...")
    
    correct_predictions = 0
    detailed_results = []
    
    for i, case in enumerate(tqdm(cases, desc="🔮 MedGemma predictions", unit="case")):
        try:
            # Load image
            image = Image.open(case['image_path']).convert('RGB')
            
            # Create prompt
            prompt = create_vqa_prompt(case['clinical_context'], case['choices'])
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Process inputs
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                generation = generation[0][input_len:]
            
            response = processor.decode(generation, skip_special_tokens=True)
            
            # Extract predicted choice
            predicted_choice_idx, predicted_answer = extract_choice_from_response(response, case['choices'])
            correct_answer = case['answer']
            
            # Check if correct
            is_correct = predicted_answer.lower().strip() == correct_answer.lower().strip()
            if is_correct:
                correct_predictions += 1
            
            # Store detailed result
            result = {
                'case_id': case['case_dir'],
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'method': 'medgemma-4b-it',
                'clinical_context': case['clinical_context'][:100] + "..." if len(case['clinical_context']) > 100 else case['clinical_context'],
                'full_response': response[:200] + "..." if len(response) > 200 else response
            }
            detailed_results.append(result)
            
            # Print progress
            status = '✅' if is_correct else '❌'
            print(f"{status} {case['case_dir']}: '{predicted_answer}' vs '{correct_answer}'")
            
        except Exception as e:
            print(f"❌ Error processing case {case['case_dir']}: {e}")
            # Add failed case
            result = {
                'case_id': case['case_dir'],
                'predicted': 'ERROR',
                'correct': case['answer'],
                'is_correct': False,
                'method': 'medgemma-4b-it',
                'clinical_context': case['clinical_context'][:100] + "..." if len(case['clinical_context']) > 100 else case['clinical_context'],
                'full_response': f'ERROR: {str(e)}'
            }
            detailed_results.append(result)
    
    accuracy = (correct_predictions / len(cases)) * 100
    return accuracy, detailed_results


def run_medgemma_evaluation():
    """Run MedGemma evaluation on NEJM chest X-ray cases"""
    print("🚀 Starting MedGemma-4B-IT Evaluation")
    print("=" * 80)
    
    # Load NEJM cases
    cases_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/multimodal_nejm/cxr_cases/"
    cases = load_nejm_cases(cases_dir)
    
    if len(cases) == 0:
        print("❌ No valid cases found!")
        return 0.0
    
    print(f"\n📊 Processing {len(cases)} valid NEJM cases")
    
    # Load MedGemma model
    model, processor = load_medgemma_model()
    
    # Evaluate
    accuracy, detailed_results = evaluate_medgemma(model, processor, cases)
    
    # Report results
    print("\n" + "=" * 80)
    print(f"📊 MEDGEMMA-4B-IT RESULTS:")
    print("=" * 80)
    print(f"Total cases processed: {len(cases)}")
    print(f"Correct predictions: {int(accuracy * len(cases) / 100)}")
    print(f"Incorrect predictions: {len(cases) - int(accuracy * len(cases) / 100)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 80)
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_file = "multimodal_nejm/medgemma_vqa_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"📁 Detailed results saved to: {results_file}")
    
    # Compare with previous results if available
    try:
        previous_results = pd.read_csv("multimodal_nejm/nejm_vqa_accuracy_summary.csv")
        print(f"\n🔍 COMPARISON WITH PREVIOUS METHODS:")
        print("-" * 50)
        for _, row in previous_results.iterrows():
            print(f"{row['method']}: {row['accuracy']:.2f}%")
        print(f"medgemma-4b-it: {accuracy:.2f}%")
        
        best_previous = previous_results['accuracy'].max()
        if accuracy > best_previous:
            improvement = accuracy - best_previous
            print(f"\n🎉 MedGemma improves by {improvement:.2f}% over best previous method!")
        else:
            deficit = best_previous - accuracy
            print(f"\n📊 MedGemma is {deficit:.2f}% below best previous method")
            
    except FileNotFoundError:
        print("📝 No previous results found for comparison")
    
    return accuracy


if __name__ == "__main__":
    accuracy = run_medgemma_evaluation()
    print(f"\n🎯 Final MedGemma Accuracy: {accuracy:.2f}%") 