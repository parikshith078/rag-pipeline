from mlx_lm import load, generate

model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
max_tokens = 140

# load model
model, tokenizer = load(model_path)

def ask_local_llm(prompt):
    response = generate(model, tokenizer, prompt=prompt, 
                                      max_tokens = max_tokens, 
                                      verbose=False)
    if isinstance(response, dict) and "response" in response:
        final_response = response["response"]
    else:
        # If `generate` returns raw text, ensure only the relevant part is extracted
        final_response = response.split("Answer:")[-1].strip()

    return final_response