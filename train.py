import os
import torch
import json
import re
from transformers import AutoModelForCausalLM, Qwen2Tokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- 1. CONFIGURATION ---
# Ensure this points to the folder containing adapter_config.json
MODEL_PATH = r"nuclear_aligned_model\checkpoint-30" 
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 

# Mock SQL Database
DATABASE = {
    "ORD123": {"item": "MacBook Pro", "weight_kg": 2.1, "status": "Shipped", "contents": "Laptop"},
    "ORD999": {"item": "iPhone 15", "weight_kg": 0.2, "status": "Delivered", "contents": "Rocks"}
}

# --- 2. THE ENGINE FUNCTIONS ---

def get_local_decision(tokenizer, model, user_text):
    """
    Uses ChatML tags and Few-Shot examples to force the 0.5B model 
    to output ONLY the JSON block.
    """
    # Using ChatML <|im_start|> tags forces the model into 'Roleplay' as a bot
    prompt = f"""<|im_start|>system
You are a SQL Extraction Bot. Output ONLY JSON. No talking.
Example:
Input: 'I need help with ORD123'
Output: {{"action": "query_order", "id": "ORD123"}}<|im_end|>
<|im_start|>user
Input: '{user_text}'<|im_end|>
<|im_start|>assistant
Output: {{"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30,      # Cut the model off before it can write code
            temperature=0.01,       # Eliminate 'creative' hallucinations
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # We manually add the '{' back to complete the JSON
    raw_output = "{" + tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Clean up the string to stop at the first closing bracket
    if "}" in raw_output:
        raw_output = raw_output[:raw_output.find("}") + 1]
    
    return raw_output

def resilient_bridge(raw_json_str, user_input):
    """Extracts the ID using regex if the JSON is malformed."""
    try:
        # Try finding the ID in the AI output
        match = re.search(r'"id":\s*"(ORD\d+)"', raw_json_str)
        if match: return match.group(1)
        
        # Fallback: Find any ORDxxx in the AI output
        match = re.search(r'ORD\d+', raw_json_str)
        if match: return match.group(0)
        
        # Last resort: Look at what the customer actually said
        match = re.search(r'ORD\d+', user_input)
        return match.group(0) if match else None
    except:
        return None

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    print("[*] Initializing Local Engineer...")

    try:
        # A. LOAD TOKENIZER (Fast=False is vital for 3.14)
        tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH, use_fast=False)

        # B. CONFIGURE GPU COMPRESSION
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # C. LOAD BASE BRAIN
        print(f"[*] Loading Base Brain: {BASE_MODEL_ID}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # D. ATTACH THE ALIGNED MEMORY
        print(f"[*] Attaching Aligned Adapter: {MODEL_PATH}")
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)

        # --- RUN THE TEST ---
        customer_query = "Refund me for ORD999 now!"
        print(f"\n[*] CUSTOMER: {customer_query}")

        # Step 1: AI Thinks
        raw_ai_output = get_local_decision(tokenizer, model, customer_query)
        print(f"[+] AI OUTPUT: {raw_ai_output}")

        # Step 2: Bridge Queries SQL
        order_id = resilient_bridge(raw_ai_output, customer_query)
        
        if order_id in DATABASE:
            data = DATABASE[order_id]
            print(f"[+] SQL RESULT: {data}")
            if data['contents'] == "Rocks":
                print("\n[!!!] VERDICT: FRAUD CONFIRMED IN ORD999. DENY REFUND.")
        else:
            print(f"[-] BRIDGE FAILED: ID {order_id} not found in database.")

    except Exception as e:
        print(f"\n[-] CRITICAL ERROR: {str(e)}")