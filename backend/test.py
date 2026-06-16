from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

print("Loading MrRoyaleAce/nyaya-7b...")

model_path = "kaggle/nyaya7b-finetuning-output/nyaya-checkpoints/nyaya-7b-merged"

tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="sdpa",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

TEST = """
IN THE SUPREME COURT OF INDIA
Criminal Appeal No. 1481 of 2018
State of Punjab v. Gurpreet Singh
Sections 302 and 34 IPC. Trial Court convicted.
High Court acquitted. Relied on AIR 1984 SC 1622.
HELD: Appeal allowed. Conviction restored.
"""

SYSTEM_PROMPT = (
    "You are Nyaya, a specialized Indian legal extraction model trained on Supreme Court "
    "and High Court judgments. Given a judgment text, extract all structured information "
    "and return it as a valid JSON object. Be precise with Indian legal citation formats "
    "(AIR, SCC, SCR). Never hallucinate statute sections or case citations not in the text."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Extract structured data from this Indian court judgment and return a JSON object:\n\n{TEST.strip()}"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

out = pipe(
    prompt,
    max_new_tokens=512,
    do_sample=False,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id,
)[0]["generated_text"].strip()

print(out)

try:
    p = json.loads(out)
    print("Valid JSON")
    print("case_name:", p.get("case_name"))
    print("outcome:", p.get("outcome"))
    print("statutes:", [s.get("section") for s in p.get("statutes_cited", [])])
except Exception as e:
    print("Not valid JSON")
    print(e)