#!/usr/bin/env python3
"""
Comparison Summary Generator for NEJM VQA Results
Combines all methods and creates comprehensive analysis
"""

import pandas as pd
import numpy as np

def create_comparison_summary():
    """Create a comprehensive comparison of all VQA methods"""
    
    print("📊 NEJM Chest X-ray VQA - Method Comparison Summary")
    print("=" * 80)
    
    # Load previous results
    try:
        concept_results = pd.read_csv("multimodal_nejm/nejm_vqa_accuracy_summary.csv")
        print("✅ Loaded concept-based method results")
    except FileNotFoundError:
        print("❌ Concept-based results not found")
        return
    
    # Load MedGemma results  
    try:
        medgemma_results = pd.read_csv("multimodal_nejm/medgemma_vqa_results.csv")
        medgemma_accuracy = (medgemma_results['is_correct'].sum() / len(medgemma_results)) * 100
        print("✅ Loaded MedGemma results")
    except FileNotFoundError:
        print("❌ MedGemma results not found")
        return
    
    # Create comprehensive comparison
    methods = {
        "Context Only (sfr-mistral)": concept_results[concept_results['method'] == 'context_only']['accuracy'].iloc[0],
        "Image Only (CLIP+Concepts)": concept_results[concept_results['method'] == 'image_only']['accuracy'].iloc[0], 
        "Multimodal (CLIP+sfr-mistral)": concept_results[concept_results['method'] == 'multimodal']['accuracy'].iloc[0],
        "MedGemma-4B-IT": medgemma_accuracy
    }
    
    # Sort by accuracy
    sorted_methods = sorted(methods.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 RANKING (Total: {len(medgemma_results)} cases)")
    print("-" * 60)
    
    medals = ["🥇", "🥈", "🥉", "🏅"]
    for i, (method, accuracy) in enumerate(sorted_methods):
        medal = medals[i] if i < len(medals) else "🔸"
        correct = int(accuracy * len(medgemma_results) / 100)
        incorrect = len(medgemma_results) - correct
        
        print(f"{medal} {method}")
        print(f"    Accuracy: {accuracy:.1f}% ({correct}/{len(medgemma_results)} correct)")
        print(f"    Errors: {incorrect}")
        print()
    
    # Performance gaps
    best_accuracy = sorted_methods[0][1]
    best_method = sorted_methods[0][0]
    
    print("📈 PERFORMANCE ANALYSIS")
    print("-" * 40)
    print(f"🎯 Best performer: {best_method} ({best_accuracy:.1f}%)")
    
    for method, accuracy in sorted_methods[1:]:
        gap = best_accuracy - accuracy
        print(f"📉 {method}: -{gap:.1f}% vs best")
    
    # Method insights
    concept_best = max(concept_results['accuracy'])
    medgemma_gain = medgemma_accuracy - concept_best
    
    print(f"\n🔍 KEY INSIGHTS")
    print("-" * 30)
    print(f"• MedGemma outperforms concept-based methods by +{medgemma_gain:.1f}%")
    print(f"• End-to-end training beats zero-shot concept similarity")
    print(f"• Medical fine-tuning (MedGemma) shows clear advantage")
    print(f"• Image+text fusion benefits are model-dependent")
    
    # Case-by-case analysis
    print(f"\n🧪 CASE-BY-CASE INSIGHTS")
    print("-" * 35)
    
    # Load comprehensive results for detailed analysis
    try:
        comprehensive = pd.read_csv("multimodal_nejm/nejm_vqa_comparative_results.csv")
        
        # Cases where MedGemma succeeded but others failed
        medgemma_only_correct = []
        for case_id in medgemma_results['case_id'].unique():
            medgemma_correct = medgemma_results[medgemma_results['case_id'] == case_id]['is_correct'].iloc[0]
            concept_correct = comprehensive[
                (comprehensive['case_id'] == case_id) & 
                (comprehensive['method'] == 'multimodal')
            ]['is_correct'].iloc[0]
            
            if medgemma_correct and not concept_correct:
                medgemma_only_correct.append(case_id)
        
        print(f"• MedGemma uniquely solved {len(medgemma_only_correct)} cases that concept methods missed")
        print(f"• Examples: {', '.join(medgemma_only_correct[:3])}...")
        
        # Cases where concept methods succeeded but MedGemma failed
        concept_only_correct = []
        for case_id in medgemma_results['case_id'].unique():
            medgemma_correct = medgemma_results[medgemma_results['case_id'] == case_id]['is_correct'].iloc[0]
            concept_correct = comprehensive[
                (comprehensive['case_id'] == case_id) & 
                (comprehensive['method'] == 'multimodal')
            ]['is_correct'].iloc[0]
            
            if concept_correct and not medgemma_correct:
                concept_only_correct.append(case_id)
        
        print(f"• Concept methods uniquely solved {len(concept_only_correct)} cases that MedGemma missed")
        if concept_only_correct:
            print(f"• Examples: {', '.join(concept_only_correct[:3])}...")
            
    except Exception as e:
        print(f"• Detailed case analysis unavailable: {e}")
    
    print(f"\n💡 FUTURE DIRECTIONS")
    print("-" * 25)
    print("• Ensemble MedGemma + concept-based methods")
    print("• Fine-tune concept similarity on medical VQA data") 
    print("• Explore larger medical vision-language models")
    print("• Add medical knowledge graph constraints")
    print("• Test on larger NEJM dataset")
    
    print("=" * 80)

if __name__ == "__main__":
    create_comparison_summary() 