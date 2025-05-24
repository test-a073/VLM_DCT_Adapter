import os
import yaml
import json
import time
from openai import OpenAI

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "openai_config.yaml")
OUTPUT_PATH = "/common/home/users/s/sasikayw/scratchDirectory/VLM_DCT_Adapter/evaluator/benchmark_datasets/new_datasamples.jsonl"
MODEL_NAME = "gpt-4o"  # or "gpt-4-turbo"
NUM_SAMPLES = 50

# 3-5 diverse MT-Bench examples from the dataset
MTBENCH_EXAMPLES = """
    # Example 1
    '{"task": "GR", "id": 1, "history": [ {"user": "Now there are three people A, B and C. I currently know that A is taller than B and B is taller than C. Who is the tallest currently?", "bot": "Based on the given information, A is the tallest among the three people."}, {"user": "Now there are two more people, D and E. D is higher than B and E is higher than D. Who is the tallest at the moment?", "bot": "Based on the additional information, it is not possible to determine who the tallest person is at the moment. The given information only states the relative heights between the individuals but does not provide any specific comparison between A, C, D, and E."}, {"user": "Now, I know that D is higher than A. Who is the highest now?", "bot": "Based on the updated information, if D is higher than A and A is already known to be taller than B and B is taller than C, E is higher than D. then E would be the tallest person at the moment."} ]}',
    # Example 2
    '{"task": "GR", "id": 6, "history": [ {"user": "I put the key inside the book and put the book on the dining table in the living room. Where is the key at this time?", "bot": "Based on the information you provided, the key is inside the book, which is placed on the dining table in the living room. That means the key is in the living room, secured within the pages of the book on the table."}, {"user": "Then I took the book to the dresser in my bedroom. Where is the key now?", "bot": "If you moved the book to the dresser in your bedroom and the key was still inside the book, then the key is now on the dresser in your bedroom, inside the book."}, {"user": "Then, I put the books into my backpack and went to school. When I came back, I put the backpack in the closet. Where are my keys now?", "bot": "Assuming the key remained inside the book and the book stayed in your backpack the whole time, your keys are now inside the backpack, which you've placed in the closet after returning from school."} ]}',
    # Example 3
    '{"task": "GR", "id": 7, "history": [ {"user": "By taking into account the facts ```fact1: Anne is not furry. fact2: Anne is rough. fact3: Anne is white. fact4: Charlie is big. Can you infer that Charlie is furry based on these facts? Please give a conclusion and explain the reasons.", "bot": "we cannot infer that Charlie is furry. None of the provided facts about Anne give us any information about Charlie's characteristics, other than the fact that Charlie is big (fact4). The characteristics of Anne and Charlie are independent of each other unless there is a stated relationship or rule connecting the two sets of characteristics, which is not provided here."}, {"user": "Okay, now add the following facts and rules, rule1: If someone is kind and big then they are smart. rule2: If someone is furry then they are kind. Can you infer that Charlie is furry based on these facts and rules? Please give a conclusion and explain the reasons.", "bot": "Based on the facts and rules provided, we cannot infer that Charlie is furry. We lack information on Charlie's kindness or smartness, which could potentially allow us to use the rules to make further inferences about Charlie's other characteristics."}, {"user": "Finally, add the following facts and rules, rule3: Smart people are furry. fact5: Charlie is kind. Can you infer that Charlie is furry based on these facts and rules now? Please give a conclusion and explain the reasons.", "bot": "Now that we know Charlie is both big and kind, we can apply rule1: Since Charlie is kind and big, rule1 allows us to infer that Charlie is also smart. Next, with the knowledge that Charlie is smart, we can apply rule3: Rule3 tells us that smart people are furry. Therefore, by combining these rules and facts, we can conclude: Charlie is kind and big (fact4 and fact5). Because Charlie is kind and big, Charlie is smart (rule1). Because Charlie is smart, Charlie is furry (rule3). In conclusion, based on the facts and rules provided, we can infer that Charlie is indeed furry."} ]}',
    # Example 4
    '{"task": "GR", "id": 8, "history": [ {"user": "I have two apples. I eat one. How many apples do I have now?", "bot": "After eating one of the two apples, you now have one apple left."}, {"user": "Then, my friend gives me two more apples. How many apples do I have in total?", "bot": "Initially, you had one apple left. After your friend gives you two more, you have a total of three apples."}, {"user": "If I eat one more apple and put one in the fridge, how many apples will I have outside the fridge?", "bot": "After eating one more apple, you'll have two apples left in total. If you put one of them in the fridge, you will have one apple remaining outside the fridge."} ]}',
    # Example 5
    '{"task": "GR", "id": 10, "history": [ {"user": "There is a red apple and a green apple on the table. I ate the red apple. Which apple is left on the table?", "bot": "If you ate the red apple, then only the green apple is left on the table."}, {"user": "If I now eat half of the green apple, what is left on the table?", "bot": "After eating half of the green apple, half of the green apple remains on the table."}, {"user": "Assuming my friend takes away the remaining half, is there any apple left on the table now?", "bot": "If your friend takes away the remaining half of the green apple, then there are no apples left on the table."} ]}',
"""

EXAMPLES_TEXT = "\n\n".join(MTBENCH_EXAMPLES)

PROMPT = (
    "Below are several examples of MT-Bench style samples. Each is a JSON object with a 'task', 'id', and a 'history' list of user/bot turns.\n"
    f"{EXAMPLES_TEXT}\n\n"
    "Now, generate a new, unique sample in the exact same format and style. "
    "The sample should be a JSON object with a new id, and a realistic, logical, multi-turn reasoning scenario. "
    "Do not copy any of the above samples. Do not include any explanations or text outside the JSON. "
    "Only output the JSON object."
)

def load_openai_key(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    key = config.get('api_key') or config.get('openai_api_key')
    base = config.get('api_base') or config.get('base_url')
    return key, base

def generate_sample(client, sample_id):
    user_prompt = PROMPT.replace('new id', f'id": {sample_id}')
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=900,
        temperature=1.0,
    )
    content = response.choices[0].message.content.strip()
    # Remove code block markers if present
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
    return content

def main():
    api_key, api_base = load_openai_key(CONFIG_PATH)
    client = OpenAI(api_key=api_key, base_url=api_base or None)
    samples = []
    ids_used = set()
    with open(OUTPUT_PATH, 'w') as fout:
        for i in range(1, NUM_SAMPLES + 1):
            for attempt in range(3):
                try:
                    print(f"Generating sample {i} (attempt {attempt+1})...")
                    sample_json = generate_sample(client, i)
                    sample = json.loads(sample_json)
                    # Ensure unique id
                    if sample.get('id') in ids_used:
                        print(f"Duplicate id {sample.get('id')}, regenerating...")
                        continue
                    ids_used.add(sample.get('id'))
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    fout.flush()
                    print(f"Sample {i} written.")
                    break
                except Exception as e:
                    print(f"Error on sample {i} attempt {attempt+1}: {e}")
                    time.sleep(2)
            else:
                print(f"Failed to generate sample {i} after 3 attempts.")
    print(f"Done. {NUM_SAMPLES} samples written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 