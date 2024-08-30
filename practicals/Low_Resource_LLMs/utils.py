import subprocess
import os
import time
import time
import pandas as pd
from tqdm import tqdm


def load_existing_data(output_filename):
    """Load existing data from the CSV file if it exists."""
    if os.path.exists(output_filename):
        existing_df = pd.read_csv(output_filename)
        return existing_df.to_dict("records"), existing_df.shape[0]
    return [], 0


def handle_rate_limit_error(wait_time):
    """Handle rate limit errors by waiting and doubling the wait time."""
    print("Rate limit error")
    print(f"Waiting for {wait_time} seconds...", flush=True)
    time.sleep(wait_time)
    return wait_time * 2


def substring_after_colon(input_string):
    colon_index = input_string.find(":")
    if colon_index != -1:
        return input_string[colon_index + 1 :]
    else:
        return input_string


def summarize_news_article(
    test_source, base_prompt, output_filename, model_name, api_key, max_train=20
):
    """
    Summarize news articles into clickbait headlines in Moroccan Darija.

    Parameters:
    - test_source: List of news articles to summarize.
    - base_prompt: list of system messages and user (few-shot) messages.
    - output_filename: Name of the output CSV file.
    - model_name: Model name to use for the OpenAI API.
    - api_key: API key for the OpenAI API.
    - max_train: Maximum number of training examples to include in prompts.
    """
    from openai import OpenAI
    import openai

    client = OpenAI(api_key=api_key)
    df_lines, existing_len = load_existing_data(output_filename)
    rewritten_prompt_count = existing_len
    wait_time = 1

    for data in tqdm(
        test_source[existing_len:], desc=f"Processing lines from {existing_len}-th line"
    ):
        news_article = data.strip()
        made_error = True
        num_error = 0
        final_prompt = base_prompt.copy()  # Step 1: Copy the base prompt
        final_prompt.append(
            {  # Step 2: Append the new dictionary
                "role": "user",
                "content": f'Summarize the following news article into a headline in Moroccan Darija only:\n"{news_article}"',
            }
        )

        while made_error:
            try:
                response = client.chat.completions.create(
                    messages=final_prompt,
                    model=model_name,
                )
                headline = response.choices[0].message.content
                df_lines.append(
                    {
                        "article": news_article,
                        "generated_headline": headline,
                        "prompt_messages": final_prompt,
                    }
                )
                rewritten_prompt_count += 1
                made_error = False

            except Exception as e:
                if isinstance(e, openai.RateLimitError):
                    wait_time = handle_rate_limit_error(wait_time)
                else:
                    print(f"Error: {e}")
                    num_error += 1
                    print("Consider Reducing the shots_count to", max_train - num_error)

    df = pd.DataFrame.from_dict(df_lines)
    df.to_csv(output_filename, index=False)
    print(f"Saved {rewritten_prompt_count} headlines to {output_filename}")


def load_models():
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EncoderDecoderModel

    # LoRA PEFT models
    config = PeftConfig.from_pretrained("alizaidi/lora-mt5-goud")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = PeftModel.from_pretrained(
        base_model, "alizaidi/lora-mt5-goud", device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained("alizaidi/lora-mt5-goud")

    lora_goud = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
    }

    # base models
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

    mt5_small = {
        "model": model,
        "tokenizer": tokenizer,
    }

    # load bert fine-tunes
    bert_finetune_names = [
        "Goud/AraBERT-summarization-goud",
        "Goud/DziriBERT-summarization-goud",
        "Goud/DarijaBERT-summarization-goud",
    ]
    bert_models = {}
    for model_name in bert_finetune_names:

        if (
            "AraBERT" in model_name
            or "DziriBERT" in model_name
            or "DarijaBERT" in model_name
        ):
            print(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = EncoderDecoderModel.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        bert_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
        }

    return lora_goud, mt5_small, bert_models


def install_requirements(requirements_path="requirements.txt"):
    process = subprocess.Popen(
        ["pip", "install", "-r", requirements_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensure the output is captured as text, not bytes
    )

    # Capture and print the output in real-time
    for line in process.stdout:
        print(line, end="")

    # Wait for the process to complete
    process.wait()

    # Check for any errors
    if process.returncode != 0:
        print("\nError during installation:")
        for line in process.stderr:
            print(line, end="")


def download_and_extract_zip(url, extract_to="."):
    import requests, zipfile, io

    print(f"Starting download from {url}...")

    # Start the download process
    response = requests.get(url, stream=True)

    # Check if the download was successful
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        print(f"Download complete. Extracting {total_size / (1024 * 1024):.2f} MB...")

        # Create a ZipFile object from the downloaded content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract all the contents of the zip file into the specified directory
            z.extractall(path=extract_to)

            # Print the names of the extracted files
            extracted_files = z.namelist()
            print(f"Extracted {len(extracted_files)} files to '{extract_to}':")
            for file in extracted_files:
                print(f" - {file}")

        print("Download and extraction complete.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
