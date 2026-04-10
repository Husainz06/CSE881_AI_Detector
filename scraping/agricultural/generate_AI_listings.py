import os
import json
import time
import random
import re
import requests
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# 2. Configuration - Using High-Capability NIM Models
MODELS = [
    "qwen/qwen2.5-7b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "meta/llama-3.1-70b-instruct",
    "openai/gpt-oss-120b",
]

LISTINGS_PER_MODEL = 100
OUTPUT_FILE = "ai_listings.json"

# 3. Dynamic Variables to force uniqueness
STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
    "Wisconsin", "Wyoming"
]

POPULATIONS = [
    "veterans experiencing PTSD or transition challenges",
    "youth at risk of homelessness",
    "individuals with developmental and intellectual disabilities",
    "people in addiction and substance use recovery",
    "formerly incarcerated individuals reentering society",
    "seniors with early-stage dementia",
    "survivors of domestic violence and trauma",
    "refugees, asylum seekers, and newly arrived immigrants",
    "children and teens with autism spectrum disorder (ASD)",
    "adults managing severe mental health conditions",
    "urban communities heavily impacted by food apartheid",
    "older teens aging out of the foster care system",
    "individuals with physical disabilities",
    "neurodivergent young adults seeking vocational skills"
]

AG_FOCUSES = [
    "organic vegetable CSA and community-supported agriculture",
    "equine-assisted therapy and livestock management",
    "orchards, fruit tree cultivation, and apiaries",
    "dairy, artisan cheesemaking, and pasture-raised cattle",
    "flower farming and herbalism",
    "regenerative poultry and agroforestry",
    "no-till farming and heavy mulching",
    "urban rooftop gardening and hydroponics",
    "mushroom cultivation and forest foraging",
    "heritage breed pigs and ethical meat production"
]

THEMES = [
    "a historic estate converted into an educational non-profit",
    "a small, family-run rural homestead",
    "a bustling urban farm built on reclaimed vacant city lots",
    "a sprawling, multi-acre cooperative farm",
    "a grassroots, volunteer-led neighborhood garden",
    "a faith-based agricultural intentional community",
]

# NEW: Instructing the LLM on exactly how to style the name
NAME_STYLES = [
    "a traditional agricultural name ending in 'Farm', 'Acres', or 'Homestead'",
    "a short, abstract, or generic conceptual name without a farm suffix (e.g., similar to 'Soulful Seeds', 'Arcadia', or 'Cultivate')",
    "a formal organizational name ending in 'LLC', 'Inc.', or 'Non-Profit'",
    "a name focused on community, ending in 'Project', 'Collaborative', or 'Initiative'",
    "a localized name referencing geographic features (e.g., 'Riverbend', 'Oak Hill') without using the word 'farm'"
]

# Updated prompt to use NAME_STYLES
PROMPT_TEMPLATE = """
You are an expert directory copywriter. Your task is to generate a single realistic, fictional profile for an agricultural organization in the United States. 
  
Content Guidelines: 
Location & Setting: Located in {state} and operates as {theme}.
Naming: The name of the organization should follow this style: {name_style}. Do not simply name it a generic "care farm".
Niche Focus: The organization integrates therapeutic agriculture or social farming, specifically serving {population}. 
Negative Constraints: DO NOT repetitively use the exact phrase "care farm". Real directories use varied terminology like "the farm", "our organization", "the project", or "our community". Avoid AI buzzwords like "delve", "tapestry", or "beacon".
Agricultural Details: The primary agricultural focus is {ag_focus}. Include specific, realistic details.
Tone and Style: Authentic, warm, and grounded. Write as if the farm owner or a local community member wrote it. 
Length: Roughly 50 to 250 words. Format the description as a single continuous string. 
 
Output Format Constraints: 
OUTPUT STRICTLY AND ONLY A VALID JSON OBJECT. Do not include any conversational text before or after the JSON.  
 
The JSON must contain the exact following keys: 
"id": a unique, lowercase, hyphen-separated, URL-friendly slug based on the listing’s name 
"url": a fictional URL formatted exactly as "https://carefarmingnetwork.org/directory-member_farms/listing/[id]/" 
"name": the fictional name 
"description": the multi-sentence paragraph
"""

def extract_json_from_text(text):
    """Uses regex to find and extract the JSON object even if the LLM added conversational text."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in the response string.")

def generate_listing(model_name):
    prompt = PROMPT_TEMPLATE.format(
        state=random.choice(STATES),
        theme=random.choice(THEMES),
        name_style=random.choice(NAME_STYLES),
        population=random.choice(POPULATIONS),
        ag_focus=random.choice(AG_FOCUSES)
    )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.85, 
        "max_tokens": 1024
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    message = result.get('choices', [{}])[0].get('message', {})
    
    if not message or not message.get('content'):
        raise ValueError("Model returned an empty response (content is None).")
        
    raw_content = message['content'].strip()
    json_string = extract_json_from_text(raw_content)
    data = json.loads(json_string)
    
    data["label"] = "AI"
    data["source_model"] = model_name 
    
    return data

def main():
    if not NVIDIA_API_KEY:
        print("Error: NVIDIA_API_KEY not found in .env environment.")
        return

    all_listings = []
    batch_request_count = 0
    seen_names = set()
    
    for model in MODELS:
        print(f"\n--- Starting generation for model: {model} ---")
        successful_count = 0
        
        while successful_count < LISTINGS_PER_MODEL:
            
            # RATE LIMITING CHECK
            if batch_request_count >= 40:
                print("\n[Rate Limit] Reached 40 requests. Sleeping for 60 seconds...")
                time.sleep(60)
                batch_request_count = 0
            
            try:
                print(f"Generating listing {successful_count + 1}/{LISTINGS_PER_MODEL} via {model}...")
                listing = generate_listing(model)
                name = listing.get('name', '').strip().lower()
                
                if name in seen_names:
                    print(f"  -> Duplicate found ({listing.get('name')}). Retrying...")
                else:
                    seen_names.add(name)
                    all_listings.append(listing)
                    successful_count += 1
                    print(f"  -> Success: {listing.get('name')}")
                    
                    if successful_count % 10 == 0:
                        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                            json.dump(all_listings, f, indent=4)
                            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  -> Parsing Error: {e}. Retrying after short delay...")
                time.sleep(2)
            except Exception as e:
                print(f"  -> Request failed: {e}. Retrying after short delay...")
                time.sleep(2) 
            
            batch_request_count += 1
            time.sleep(3)

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_listings, f, indent=4)
        
    print(f"\nDone! Successfully generated and saved {len(all_listings)} unique listings to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()