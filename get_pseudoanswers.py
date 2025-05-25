import json
import ast
import numpy as np
from tqdm import tqdm
from openai import AzureOpenAI

def get_concepts(file_path, pathology_exclude=None, subsample=0.1):
    with open(file_path, 'r') as f:
        results = json.load(f)

    concepts = []
    for result in tqdm(results):
        try:
            re_dict = ast.literal_eval(result['model_output'])
            concepts.extend(re_dict['observations'])
        except:
            continue
    concepts = [concept.lower() for concept in concepts]
    concepts = list(set(concepts))
    
    if pathology_exclude:
        print(f"Total concepts before exclusion: {len(concepts)}")
        concepts = [concept for concept in concepts if pathology_exclude not in concept]  
        print(f"Total concepts after excluding '{pathology_exclude}': {len(concepts)}")

    if subsample:
        np.random.seed(42)
        subsample_size = int(len(concepts) * subsample)
        concepts = np.random.choice(concepts, size=subsample_size, replace=False).tolist()
    
    print(f"Total concepts: {len(concepts)}")
    
    return concepts


if __name__ == "__main__":
    file_path = 'data/mimic_concepts.json'
    pathology_exclude = 'cardiomegaly'
    subsample = 0.1
    concepts = get_concepts(file_path, pathology_exclude, subsample)
    

    client = AzureOpenAI(
        azure_endpoint = "https://ukatrki02.openai.azure.com", 
        api_key="235fa0c2e26b4595aca8227923c59720", 
        api_version="2025-04-01-preview",
        )
    
    batch_size = 100
    results = {}
    for i in tqdm(range(0, len(concepts), batch_size)):
        batch = concepts[i:i + batch_size]
        prompt = f"""
You are a helpful assistant. Please respond in valid JSON only.
Question: Answer yes/no/depends for whether the following concepts are clinically relevant for diagnosing '{pathology_exclude}': {batch}
Output format: <concept>: <answer>.
"""
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        ).choices[0].message.content

        try:
            json_response = json.loads(response)
            results.update(json_response)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from response: {response}")
        # Save results to a JSON file
        with open(f'data/mimic_pseudoans_{pathology_exclude}_{subsample}.json', 'w') as f:
            json.dump(results, f, indent=4)
