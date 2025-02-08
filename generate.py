from unsloth import FastLanguageModel
from vllm import SamplingParams
import argparse
import torch

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def load_model(
    model_path: str,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
    gpu_memory_utilization: float = 0.5
):
    # Load the base model and tokenizer with LoRA weights
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        gpu_memory_utilization=gpu_memory_utilization,
        device_map="cuda:0"
    )
    model = FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_response(
    prompt: str,
    model: FastLanguageModel,
    tokenizer,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_output_tokens: int = 1024
) -> str:
    # Prepare the messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # TODO Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens
    )
    del sampling_params

    input_tensor = tokenizer.encode(text, return_tensors="pt").to("cuda")
    
    # Generate the response
    output = model.generate(
        input_tensor,
        temperature=temperature,
        top_p=top_p,
        # Could be max_new_tokens or max_length for total
        max_new_tokens=max_output_tokens
        # sampling_params=sampling_params
    )
    # Decode the output tokens to text
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return decoded_output

def main():
    parser = argparse.ArgumentParser(description="Generate responses using a fine-tuned model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint, such as saved_model/")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a CUDA-capable GPU and drivers installed.")
    
    # Load the model
    print("Loading model...")
    model, tokenizer = load_model(args.model_path)
    
    # Generate response
    print("\nGenerating response...\n")
    response = generate_response(
        args.prompt,
        model,
        tokenizer,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens
    )
    
    print("Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    main() 