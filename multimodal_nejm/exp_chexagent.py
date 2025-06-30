#!/usr/bin/env python3
"""
CheXagent-8b Evaluation for NEJM Chest X-ray VQA Cases
Using Stanford's medical multimodal model for comparison
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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

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


def load_chexagent_model():
    """Load CheXagent-8b model"""
    print("🤖 Loading CheXagent-8b model...")
    
    model_name = "StanfordAIMI/CheXagent-8b"
    dtype = torch.float16
    device = "cuda"
    
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map="auto"  # Automatically handle device placement
        )
        model.eval()
        
        print("✅ CheXagent-8b model loaded successfully")
        return model, processor, generation_config
    
    except Exception as e:
        print(f"❌ Error loading CheXagent model: {e}")
        print("💡 Make sure you have access to StanfordAIMI/CheXagent-8b")
        raise


def create_vqa_prompt(clinical_context: str, choices: List[str]) -> str:
    """Create a VQA prompt for CheXagent"""
    prompt = f"""Based on the chest X-ray image and clinical context, select the most appropriate answer.

Clinical Context: {clinical_context}

Choose from the following options:
"""
    
    for i, choice in enumerate(choices, 1):
        prompt += f"{i}. {choice}\n"
    
    prompt += "\nRespond with only the number and text of your chosen answer."
    
    return prompt


def extract_choice_from_response(response: str, choices: List[str]) -> Tuple[int, str]:
    """Extract the chosen answer from CheXagent's response"""
    response = response.strip().lower()
    
    # Remove common prefixes that might appear in the response
    response = re.sub(r'^(assistant:|user:|\s)*', '', response, flags=re.IGNORECASE)
    
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


def evaluate_chexagent(model, processor, generation_config, cases: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluate CheXagent on NEJM VQA cases"""
    print(f"🎯 Evaluating CheXagent-8b on {len(cases)} cases...")
    
    device = "cuda"
    dtype = torch.float16
    correct_predictions = 0
    detailed_results = []
    
    for i, case in enumerate(tqdm(cases, desc="🔮 CheXagent predictions", unit="case")):
        try:
            # Create prompt
            prompt = create_vqa_prompt(case['clinical_context'], case['choices'])
            
            # Load image
            image = Image.open(case['image_path']).convert("RGB")
            images = [image]
            
            # Use CheXagent-8b inference format
            inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt")
            
            # Move inputs to same device as model (with device_map="auto", model handles device placement)
            if hasattr(model, 'device'):
                device = model.device
            else:
                device = next(model.parameters()).device
            
            # Move inputs to correct device and dtype
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=generation_config)[0]
            
            response = processor.tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract predicted choice
            predicted_choice_idx, predicted_answer = extract_choice_from_response(response, case['choices'])
            correct_answer = case['answer']
            
            # Check if correct
            is_correct = predicted_answer.lower().strip() == correct_answer.lower().strip()
            if is_correct:
                correct_predictions += 1
            
            # Store simplified result
            result = {
                'case_id': case['case_dir'],
                'predicted_option': predicted_answer,
                'ground_truth_option': correct_answer,
                'correct': is_correct
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
                'predicted_option': 'ERROR',
                'ground_truth_option': case['answer'],
                'correct': False
            }
            detailed_results.append(result)
    
    accuracy = (correct_predictions / len(cases)) * 100
    return accuracy, detailed_results


def run_chexagent_evaluation():
    """Run CheXagent evaluation on NEJM chest X-ray cases"""
    print("🚀 Starting CheXagent-8b Evaluation")
    print("=" * 80)
    
    # Load NEJM cases
    cases_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/multimodal_nejm/cxr_cases/"
    cases = load_nejm_cases(cases_dir)
    
    if len(cases) == 0:
        print("❌ No valid cases found!")
        return 0.0
    
    print(f"\n📊 Processing {len(cases)} valid NEJM cases")
    
    # Load CheXagent model
    model, processor, generation_config = load_chexagent_model()
    
    # Evaluate
    accuracy, detailed_results = evaluate_chexagent(model, processor, generation_config, cases)
    
    # Report results
    print("\n" + "=" * 80)
    print(f"📊 CHEXAGENT-8B RESULTS:")
    print("=" * 80)
    print(f"Total cases processed: {len(cases)}")
    print(f"Correct predictions: {int(accuracy * len(cases) / 100)}")
    print(f"Incorrect predictions: {len(cases) - int(accuracy * len(cases) / 100)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 80)
    
    # Save results to results/ directory
    os.makedirs("multimodal_nejm/results", exist_ok=True)
    results_df = pd.DataFrame(detailed_results)
    results_file = "multimodal_nejm/results/chexagent_vqa_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"📁 Results saved to: {results_file}")
    
    # Compare with previous results if available
    try:
        previous_results = pd.read_csv("multimodal_nejm/nejm_vqa_accuracy_summary.csv")
        print(f"\n🔍 COMPARISON WITH PREVIOUS METHODS:")
        print("-" * 50)
        for _, row in previous_results.iterrows():
            print(f"{row['method']}: {row['accuracy']:.2f}%")
        print(f"chexagent-8b: {accuracy:.2f}%")
        
        best_previous = previous_results['accuracy'].max()
        if accuracy > best_previous:
            improvement = accuracy - best_previous
            print(f"\n🎉 CheXagent-8b improves by {improvement:.2f}% over best previous method!")
        else:
            deficit = best_previous - accuracy
            print(f"\n📊 CheXagent-8b is {deficit:.2f}% below best previous method")
            
    except FileNotFoundError:
        print("📝 No previous results found for comparison")
    
    # Compare with MedGemma if available
    try:
        medgemma_results = pd.read_csv("multimodal_nejm/results/medgemma_vqa_results.csv")
        medgemma_accuracy = (medgemma_results['correct'].sum() / len(medgemma_results)) * 100
        print(f"\n🆚 COMPARISON WITH MEDGEMMA:")
        print("-" * 35)
        print(f"MedGemma-4B-IT: {medgemma_accuracy:.2f}%")
        print(f"CheXagent-8b: {accuracy:.2f}%")
        
        if accuracy > medgemma_accuracy:
            improvement = accuracy - medgemma_accuracy
            print(f"🎉 CheXagent-8b outperforms MedGemma by {improvement:.2f}%!")
        else:
            deficit = medgemma_accuracy - accuracy
            print(f"📊 CheXagent-8b is {deficit:.2f}% below MedGemma")
            
    except FileNotFoundError:
        print("📝 No MedGemma results found for comparison")
    
    # Compare with previous CheXagent-2-3b if available
    try:
        chexagent2_results = pd.read_csv("multimodal_nejm/results/chexagent2_vqa_results.csv")
        chexagent2_accuracy = (chexagent2_results['correct'].sum() / len(chexagent2_results)) * 100
        print(f"\n🆚 COMPARISON WITH CHEXAGENT-2-3B:")
        print("-" * 35)
        print(f"CheXagent-2-3b: {chexagent2_accuracy:.2f}%")
        print(f"CheXagent-8b: {accuracy:.2f}%")
        
        if accuracy > chexagent2_accuracy:
            improvement = accuracy - chexagent2_accuracy
            print(f"🎉 CheXagent-8b outperforms CheXagent-2-3b by {improvement:.2f}%!")
        else:
            deficit = chexagent2_accuracy - accuracy
            print(f"📊 CheXagent-8b is {deficit:.2f}% below CheXagent-2-3b")
            
    except FileNotFoundError:
        print("📝 No previous CheXagent-2-3b results found for comparison")
    
    return accuracy


if __name__ == "__main__":
    accuracy = run_chexagent_evaluation()
    print(f"\n🎯 Final CheXagent-8b Accuracy: {accuracy:.2f}%") 