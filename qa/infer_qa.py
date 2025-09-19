import base64
import json
from openai import OpenAI
from PIL import Image
import io 
import os
import logging
from multiprocessing import Pool
import time
import re
from tqdm import tqdm
import argparse

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


sys_prompt = """You are a helpful assistant."""

qa_prefix = "Based on the images above and the information extracted from it, please answer the following question:\n\n"
qa_prompt = "\n\nProvide your final answer using LaTeX in the format $$\\boxed{}$$"


def send_file_request(url):
    try:
        if not os.path.exists(url):
            logger.error(f"File not found: {url}")
            return None
        
        pil_img = Image.open(url).convert("RGB")
        min_size = 28
        width, height = pil_img.size
        if width < min_size or height < min_size:
            new_width = max(width, min_size)
            new_height = max(height, min_size)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"
    
    except Exception as e:
        logger.error(f"Error processing image {url}: {e}")
        return None

def encode_base64_from_local_file(filepath: str) -> str:
    """Encode a local file to base64 format."""
    with open(filepath, "rb") as f:
        return f"""data:image/jpeg;base64,{base64.b64encode(f.read()).decode("utf-8")}"""  


def run_single_qa(url: str, question: str, client: OpenAI, model: str) -> str:
    try:
        image_urls = []
        image_url_0 = url
        url_local = url.replace("global", "local")
        image_url_1 = url_local
        
        image_urls = [
            {"view": "global", "path": image_url_0, "prefix": "Global view: "},
            {"view": "local", "path": image_url_1, "prefix": "Local view: "},
        ]
        
        user_conv = []
        for img in image_urls:
            if os.path.exists(img["path"]):
                bs4 = send_file_request(img["path"])
                user_conv.append({"type": "image_url", "image_url": {"url": bs4}})
                
        user_conv.append({"type": "text", "text": f"""{qa_prefix}{question}'\n'{qa_prompt}"""})
        if not image_urls:
            return ""
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": user_conv,
            },
        ]
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=128,
            temperature=0.6,
            top_p=0.2,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error processing image {url}: {str(e)}")
        return f""


def process_qa_item(item, client, model, tmp_file, test_image_dir):
    try: 
        img_url = item["qa_paths"]
        if img_url.startswith("test_processed_anno"):
            img_url = img_url.replace("test_processed_anno/", "")
            img_url = os.path.join(test_image_dir, img_url)
    
        question = item["qa_questions"]
        answer = run_single_qa(img_url, question, client, model)
        choices = re.search(r"\$\$\\boxed\{([A-Za-z0-9]+)\}\$\$", answer).group(1)
        
        with open(f'{tmp_file}', 'a', encoding='utf-8') as f:
            json.dump({'id': item["qa_ids"], 'correct': choices}, f, ensure_ascii=False)
            f.write("\n")
            
        return {'id': item["qa_ids"], 'correct': choices}
    
    except Exception as e:
        logger.error(f"Error processing index {item}: {e}")
        return {'id': item["qa_ids"], 'correct': "Error"}


def main():
    parser = argparse.ArgumentParser(description="Run VQA pipeline with local model & OpenAI API")
    parser.add_argument("--test_image_dir", type=str, required=True, help="Path to processed test images")
    parser.add_argument("--test_file_dir", type=str, required=True, help="Path to test file dir containing JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Final output JSON file")
    parser.add_argument("--tmp_file", type=str, required=True, help="Temporary JSON file for progress saving")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--openai_api_base", type=str, required=True, help="OpenAI API base URL")
    parser.add_argument("--model", type=str, required=True, help="Path or name of model to use")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for multiprocessing")
    args = parser.parse_args()

    start_time = time.time()
    
    with open(f'{args.test_file_dir}/extracted_questions_4.json', 'r', encoding='utf-8') as file:
        question_data = json.load(file)
    
    assert args.tmp_file != args.output_file
    
    dict_infer = [{ "qa_ids": q['id'],
                    "qa_questions": q['text'], 
                    "qa_paths": q['path'],
                    "qa_labels": int(q['labels'][0]),
                    "qa_folders": q['folder'],
    } for q in question_data
    ]
    
    if os.path.isfile(args.tmp_file):
        with open(args.tmp_file, "r") as f:
            processed_data = [json.loads(line) for line in f]
        logger.info(f"Finished processing {len(processed_data)} items in tmp file.")
    else:
        processed_data = None

    if processed_data:
        processed_id = [i["id"] for i in processed_data]
        dict_infer = [i for i in dict_infer if i["qa_ids"] not in processed_id]
    
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base)

    if dict_infer:
        with Pool(processes=args.num_processes) as pool:
            results = list(tqdm(
                pool.imap(lambda x: process_qa_item(x, client, args.model, args.tmp_file, args.test_image_dir), dict_infer),
                total=len(dict_infer)
            ))
    else:
        results = []
        
    if processed_data:
        results += processed_data   
    
    with open(f'{args.output_file}', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
                       
    logger.info(f"Finished processing {len(results)} items in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
