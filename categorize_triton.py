import json
import os
import random
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from urllib.parse import quote

import anthropic
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

import plotly.graph_objects as go
import tomli
import umap
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_triton_functions(file_path: str) -> List[Dict]:
    """Load triton functions from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_prompts(toml_path: str) -> Dict:
    """Load categorization prompt from TOML file."""
    prompts = {}
    with open(toml_path, "rb") as f:
        config = tomli.load(f)
    prompts_toml = config["prompts"]
    for p in prompts_toml:
        prompts[p["name"]] = p["prompt"]
    if len(prompts) != len(prompts_toml):
        raise ValueError(f"Some prompts have the same name in {toml_path}")
    return prompts


def batch_items(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Split items into batches."""

    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def categorize_batch(function_batch: List[Dict], client: anthropic.Client, prompt: str, attempt: int = 0, gt_categories: List[str] = [], temperature: float = 0.2) -> List[Dict]:
    """Categorize a batch of triton functions using Claude."""
    MAX_ATTEMPTS = 10
    results = []
    # print(f"gt categories: {gt_categories}")
    if attempt > MAX_ATTEMPTS:
        raise Exception(f"Failed to categorize batch after {MAX_ATTEMPTS} attempts")
    batch = {}
    total_functions = len(function_batch)
    for i in range(0, total_functions):
        batch[i] = function_batch[i]["input"]

    input_for_prompt = [
        {"uuid": uuid, "function": function} for uuid, function in batch.items()
    ]

    try:
        # Create a message that includes the function and prompt
        message = f"{prompt}\n\n<triton_functions>\n{json.dumps(input_for_prompt)}\n</triton_functions>"

        # Call Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=temperature,
            system="You are a helpful assistant that categorizes Triton functions. Always respond with valid JSON.",
            messages=[{"role": "user", "content": message}],
        )
        # Parse the response
        results = json.loads(response.content[0].text)
        validate_schema(results)
        # Add result to our list
        if len(results) != total_functions:
            raise Exception(f"Number of returned functions does not match number of input functions. {len(results)} != {total_functions}")
        category_subcategories = [(item['category'], item['subcategory']) for item in results]
        uuids = [item['uuid'] for item in results]
        # check if all hashes are in results
        uuids_gt = [i for i in range(0, total_functions)]
        num_uuids_in_results = len([h for h in uuids_gt if h in uuids])
        if num_uuids_in_results != total_functions:
            raise Exception(f"Only {num_uuids_in_results} are in the results out of {total_functions}")
        for item in results:
            category = item['category']
            subcategory = item['subcategory']
            if category not in list(gt_categories.keys()):
                raise Exception(f"Category {category} is not in the ground truth categories whcih are {gt_categories}")
            if subcategory == "":
                continue
            if subcategory not in gt_categories[category]:
                if attempt < MAX_ATTEMPTS * 2/3:
                    raise Exception(f"Subcategory {subcategory} is not in the ground truth subcategories for category {category} which are {gt_categories[category]}")
                else:
                    # if we can't categorize it then just leave it as uncategorized 
                    item['subcategory'] = ""
    except Exception as e:
        print(f"Error processing: {str(e)}")
        # try again to caetegorize the batch
        print(f"Trying to categorize {len(batch)} functions again at attempt {attempt}")
        time.sleep(5)
        # shuffle the batch
        return categorize_batch(function_batch, client, prompt, attempt + 1, gt_categories=gt_categories, temperature=temperature+0.05)
    
    # format data
    ret = []
    for data in results:
        ret.append({
            "input": batch[data["uuid"]],
            "category": data["category"],
            "subcategory": data["subcategory"]
        })

    return ret


def get_categories(
    examples: List[Dict], client: anthropic.Client, prompt: str
) -> List[str]:
    """Get categories from a list of examples."""
    examples = [item["input"] for item in examples]
    try:
        # Create a message that includes the function and prompt
        message = f"{prompt}\n\n{json.dumps(examples)}"

        # Call Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=0,
            system="You are a helpful assistant that categorizes Triton functions. Always respond with valid JSON.",
            messages=[{"role": "user", "content": message}],
        )
        # Parse the response
        category_data = json.loads(response.content[0].text)

        # Add result to our list
        results = category_data
        
    except Exception as e:
        print(f"Error categorizing triton functions: {e}")
    return results


def create_category_chart(
    categorized_data: List[Dict], output_path: str = "category_distribution.png"
):
    """Create a bar chart of category distribution."""
    # Count categories (values)
    category_counts = defaultdict(int)

    for item in categorized_data:
        category_counts[item["category"]] += 1

    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(category_counts.keys(), category_counts.values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribution of Triton Function Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def validate_schema(categories_json: List[Dict]):
    # check for uuid, category, subcategory, keys
    for item in categories_json:
        if 'uuid' not in item:
            raise ValueError(f"UUID is not in {item}")
        if 'category' not in item:
            raise ValueError(f"Category is not in {item}")
        if 'subcategory' not in item:
            raise ValueError(f"Subcategory is not in {item}")
    


def make_folders(categories_json: List[Dict]):
    for function_dict in categories_json:
        function = function_dict['input']
        category = function_dict['category']
        uuid = function_dict['uuid']
        subcategory = function_dict['subcategory']
        if subcategory == "":
            subcategory = "uncategorized"
        os.makedirs(f"sorted_functions/{category}", exist_ok=True)
        os.makedirs(f"sorted_functions/{category}/{subcategory}", exist_ok=True)
        
        # write function as a unique file in the category folder
        with open(os.path.join(f"sorted_functions/{category}/{subcategory}", str(uuid) + '.py'), 'w') as f:
            # write imports
            f.write("import triton\n")
            f.write("import triton.language as tl\n")
            f.write("import torch\n\n")
            f.write(function)

def get_embeddings(functions_input: List[Dict]):
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    embeddings = []
    # get list of functions and uuids in same order
    functions = [function_dict["input"] for function_dict in functions_input]
    uuids = [function_dict["uuid"] for function_dict in functions_input]
    categories = [function_dict["category"] for function_dict in functions_input]
    subcategories = [function_dict["subcategory"] for function_dict in functions_input]
    # truncate both to 10 inputs
    embeddings = model.encode(functions, show_progress_bar=True)
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    return embeddings, uuids, categories, subcategories


def create_visualisation(uuids, categories, subcategories, embeddings):
    # Set random seed for reproducibility
    np.random.seed(110)

    # Apply UMAP with improved parameters and random seed
    print("Applying UMAP...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=20,
        min_dist=0.4,
        metric="cosine",
        random_state=110,  # Added fixed random state
        spread=2.0,
        n_epochs=750,
        repulsion_strength=2.0,
    )
    coords = reducer.fit_transform(embeddings)

    # Define a color palette
    color_palette = px.colors.qualitative.Set3[:8]  # Using plotly's Set3 palette
    category_colors = {
        cat: color_palette[i % len(color_palette)]
        for i, cat in enumerate(sorted(set(categories)))
    }

    # Create traces for each category
    traces = []
    for category in sorted(set(categories)):
        indices = [i for i, cat in enumerate(categories) if cat == category]
        trace = go.Scatter3d(
            x=coords[indices, 0],
            y=coords[indices, 1],
            z=coords[indices, 2],
            mode="markers",
            name=category,
            hovertext=[
                f'Title: {uuids[i]}<br>Category: {category}<br>Subcategory: {subcategories[i]}<br><a href="https://github.com/PaliC/categorize_triton_functions/blob/main/sorted_functions/{quote(category)}/{uuids[i]}.py">https://github.com/PaliC/categorize_triton_functions/blob/main/sorted_functions/{quote(category)}/{uuids[i]}.py</a>'
                for i in indices
            ],
            hoverinfo="text",
            marker=dict(
                size=8,
                opacity=0.65,
                color=category_colors[category],
                line=dict(width=1, color="white"),
                symbol="circle",
            ),
        )
        traces.append(trace)
    # Create figure with improved layout
    fig = go.Figure(data=traces)

    # Update layout with improved parameters
    fig.update_layout(
        title={
            "text": "Triton Functions 3D Visualisation (UMAP)",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=24),
        },
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2.0, y=2.0, z=1.8),
            ),
            aspectmode="data",
        ),
        width=2400,
        height=1600,
        showlegend=True,
        legend=dict(
            title=dict(text="Function Categories", font=dict(size=16)),
            itemsizing="constant",
            itemwidth=30,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig

def main():
    # Configuration
    BATCH_SIZE = 15
    CATEGORY_SAMPLE_SIZE = 150
    # BATCH_SIZE = 5
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
    # add categoreis to categrize_text prompt and subcategories
    categorize_prompt = categorize_prompt.replace("[[CATEGORIES]]", json.dumps(list(categories.keys())))
    categorize_prompt = categorize_prompt.replace("[[SUBCATEGORIES_DICT]]", json.dumps(categories))

    # Create batches
    batches = batch_items(triton_functions, BATCH_SIZE)
    
    # Process all batches
    all_results = []
    time.sleep(60)
    for batch in tqdm(batches, desc="Processing batches"):
        # wait 10 seconds before processing next batch
        time.sleep(5)
        batch_results = categorize_batch(batch, client, categorize_prompt, gt_categories=categories)
        all_results.extend(batch_results)
        print(batch_results)

    for result in all_results:
        result["uuid"] = str(uuid.uuid4())

    with open('categorized_functions.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create visualization
    create_category_chart(all_results)

    # make folders
    make_folders(all_results)

if __name__ == "__main__":
    # main()
    with open("categorized_functions.json") as f:
        categories_json = json.load(f)
    embeddings, uuids, categories, subcategories = get_embeddings(categories_json)
    fig = create_visualisation(uuids=uuids, categories=categories, subcategories=subcategories, embeddings=embeddings)
    # fig.show()

    # Save to HTML with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(f"triton_visualisation_umap_{timestamp}.html")
    fig.write_html(output_path)
    print(f"Visualisation saved to: {output_path}")
