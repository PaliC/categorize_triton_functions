import json
import tomli
import anthropic
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict
import os
from tqdm import tqdm
import dotenv
import random
import time
import uuid

def load_triton_functions(file_path: str) -> List[Dict]:
    """Load triton functions from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_prompts(toml_path: str) -> Dict:
    """Load categorization prompt from TOML file."""
    prompts = {}
    with open(toml_path, 'rb') as f:
        config = tomli.load(f)
    prompts_toml = config['prompts']
    for p in prompts_toml:
        prompts[p['name']] = p['prompt']
    if len(prompts) != len(prompts_toml):
        raise ValueError(f"Some prompts have the same name in {toml_path}")
    return prompts

def batch_items(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def categorize_batch(function_batch: List[Dict], client: anthropic.Client, prompt: str, attempt: int = 0, gt_categories: List[str] = []) -> List[Dict]:
    """Categorize a batch of triton functions using Claude."""
    results = []
    # print(f"gt categories: {gt_categories}")
    if attempt > 10:
        raise Exception("Failed to categorize batch after 10 attempts")
    
    total_functions = len(function_batch)
    batch = {}
    for i in range(0, total_functions):
        batch[i] = function_batch[i]["input"]
    
    input_for_prompt = [{"uuid": uuid, "function": function} for uuid, function in batch.items()]

    try:
        # Create a message that includes the function and prompt
        message = f"{prompt}\n\n<triton_functions>\n{json.dumps(input_for_prompt)}\n</triton_functions>"
        
        # Call Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=0.2,
            system="You are a helpful assistant that categorizes Triton functions. Always respond with valid JSON.",
            messages=[{"role": "user", "content": message}]
        )
        # Parse the response
        category_data = json.loads(response.content[0].text)
        
        # Add result to our list
        results = category_data
        if len(results) != total_functions:
            raise Exception(f"Number of returned functions does not match number of input functions. {len(results)} != {total_functions}")
        categories = [item['category'] for item in results]
        uuids = [item['uuid'] for item in results]
        # check if all hashes are in results
        uuids_gt = [i for i in range(0, total_functions)]
        num_uuids_in_results = len([h for h in uuids_gt if h in uuids])
        if num_uuids_in_results != total_functions:
            raise Exception(f"Only {num_uuids_in_results} are in the results out of {total_functions}")
        for category in categories:
            if category not in gt_categories:
                raise Exception(f"Category {category} is not in the ground truth categories whcih are {gt_categories}")
        
        
    except Exception as e:
        print(f"Error processing: {str(e)}")
        # try again to caetegorize the batch
        print(f"Trying to categorize {len(batch)} functions again at attempt {attempt}")
        time.sleep(30)
        # shuffle the batch
        return categorize_batch(function_batch, client, prompt, attempt + 1, gt_categories=gt_categories)
    
    # remove hashes from results and replace with function
    # make results a dict of function to category
    results = {batch[int(item['uuid'])]: item['category'] for item in results}
    
    return results

def get_categories(examples: List[Dict], client: anthropic.Client, prompt: str) -> List[str]:
    """Get categories from a list of examples."""
    examples = [item['input'] for item in examples]
    try:
        # Create a message that includes the function and prompt
        message = f"{prompt}\n\n{json.dumps(examples)}"
        
        # Call Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=0,
            system="You are a helpful assistant that categorizes Triton functions. Always respond with valid JSON.",
            messages=[{"role": "user", "content": message}]
        )
        # Parse the response
        category_data = json.loads(response.content[0].text)
        
        # Add result to our list
        results = category_data["categories"]
        
    except Exception as e:
        print(f"Error categorizing triton functions: {e}")
    return results

def create_category_chart(categorized_data: List[Dict], output_path: str = "category_distribution.png"):
    """Create a bar chart of category distribution."""
    # Count categories (values)
    category_counts = defaultdict(int)

    for item in categorized_data:
        category_counts[item['category']] += 1
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(category_counts.keys(), category_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Triton Function Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def format_results(results: Dict) -> List[Dict]:
    final_json = []
    for function, category in results.items():
        final_json.append({"input": function, "category": category, "uuid": str(uuid.uuid4())})
    return final_json

def make_folders(categories_json: List[Dict]):
    for function_dict in categories_json:
        function = function_dict['input']
        category = function_dict['category']
        uuid = function_dict['uuid']
        os.makedirs(f"sorted_functions/{category}", exist_ok=True)
        
        # write function as a unique file in the category folder
        with open(os.path.join(f"sorted_functions/{category}", str(uuid) + '.py'), 'w') as f:
            # write imports
            f.write("import triton\n")
            f.write("import triton.language as tl\n")
            f.write("import torch\n\n")
            f.write(function)

def main():
    # Configuration
    BATCH_SIZE = 20
    CATEGORY_SAMPLE_SIZE = 100
    # CATEGORY_SAMPLE_SIZE = 15
    # get anthropic api key from .env file
    dotenv.load_dotenv()
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    
    if not ANTHROPIC_API_KEY:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    # Initialize Anthropic client
    client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
    
    # Load data and prompt
    triton_functions = load_triton_functions('github_triton.json')
    random.shuffle(triton_functions)
    examples = triton_functions[:CATEGORY_SAMPLE_SIZE]

    prompts = load_prompts('prompts.toml')
    categorize_prompt = prompts['categorize_functions']
    get_categories_prompt = prompts['get_categories']
    categories = get_categories(examples, client, get_categories_prompt)
    print(categories)
    # add categoreis to categrize_text prompt
    categorize_prompt = categorize_prompt.replace("[[CATEGORIES]]", json.dumps(categories))

    # Create batches
    batches = batch_items(triton_functions, BATCH_SIZE)
    
    # Process all batches
    all_results = {}
    time.sleep(60)
    for batch in tqdm(batches, desc="Processing batches"):
        # wait 10 seconds before processing next batch
        time.sleep(10)
        batch_results = categorize_batch(batch, client, categorize_prompt, gt_categories=categories)
        all_results.update(batch_results)
    
    final_json = format_results(all_results)

    with open('categorized_functions.json', 'w') as f:
        json.dump(final_json, f, indent=2)

    # Create visualization
    create_category_chart(final_json)

if __name__ == "__main__":
    # main()
    categories_json = json.load(open('categorized_functions.json'))
    make_folders(categories_json)
