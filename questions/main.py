import yaml
import json
import os
from openai import OpenAI
import base64
from uuid import uuid4

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def load_api_key(config_path="openai_config.yaml"):
    """Loads the OpenAI API key from a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get("openai_api_key")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading API key from '{config_path}': {e}")
        return None

def generate_image_questions_and_answers(image_path, api_key, target_num_questions=50):
    """
    Generates a target number of questions and answers about an image using OpenAI,
    making multiple API calls if necessary.
    """
    client = OpenAI(api_key=api_key)
    all_questions_data = []
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    batch_size = 25 # Number of questions to request per API call

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        print("Error: Image encoding failed. Cannot proceed.")
        return all_questions_data

    print(f"Attempting to generate {target_num_questions} questions for image: {image_path} using OpenAI...")

    while len(all_questions_data) < target_num_questions:
        questions_needed_this_call = min(batch_size, target_num_questions - len(all_questions_data))
        if questions_needed_this_call <= 0: # Should not happen if loop condition is correct, but as a safeguard
            break

        print(f"Making API call to generate {questions_needed_this_call} more questions...")

        prompt_text = (
            f"You are an assistant tasked with generating educational questions about an image. "
            f"Given the image, generate exactly {questions_needed_this_call} diverse questions about its content, "
            "the scene, objects present, potential actions, and any reasonable inferences that can be made. "
            "For each question, provide a concise and accurate answer based on the image. "
            "Your response MUST be a JSON formatted list of objects. Each object in the list must "
            "contain exactly two keys: \"question\" and \"answer\". Do not include any introductory text, "
            "explanations, or any content outside of this JSON list. Ensure the questions are varied and cover different aspects of the image."
        )
        
        raw_response_content = ""
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=200 + (questions_needed_this_call * 150) # Estimate tokens: base + per Q&A pair (adjust as needed)
            )

            raw_response_content = response.choices[0].message.content
            
            if raw_response_content.strip().startswith("```json"):
                raw_response_content = raw_response_content.strip()[7:-3].strip()
            elif raw_response_content.strip().startswith("```"):
                raw_response_content = raw_response_content.strip()[3:-3].strip()

            parsed_qa_list = json.loads(raw_response_content)

            if not isinstance(parsed_qa_list, list):
                print("Error: API response is not a list as expected. Stopping further API calls.")
                break # Stop if the format is wrong

            processed_this_call = 0
            for qa_pair in parsed_qa_list:
                if len(all_questions_data) >= target_num_questions:
                    break # Stop adding if we've already reached the target
                if 'question' in qa_pair and 'answer' in qa_pair:
                    all_questions_data.append({
                        "question_id": str(uuid4()),
                        "question": qa_pair["question"],
                        "image_id": image_id,
                        "answer": qa_pair["answer"]
                    })
                    processed_this_call += 1
                else:
                    print(f"Warning: Skipping an item from API response due to missing 'question' or 'answer': {qa_pair}")
            
            print(f"Successfully processed {processed_this_call} Q/A pairs from this API response.")
            print(f"Total questions collected so far: {len(all_questions_data)}/{target_num_questions}")

            if processed_this_call == 0 and questions_needed_this_call > 0:
                print("Warning: API call returned 0 questions when some were expected. Stopping to avoid potential infinite loop.")
                break

        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON response from OpenAI: {e}. Stopping further API calls.")
            print(f"Raw response content was: {raw_response_content}")
            break 
        except openai.APIError as e:
            print(f"Error from OpenAI API: {e}. Stopping further API calls.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Stopping further API calls.")
            break

    # Truncate if we overshot, though the inner loop should prevent this for question adding.
    final_questions_data = all_questions_data[:target_num_questions]
    
    if len(final_questions_data) < target_num_questions:
        print(f"Warning: Collected {len(final_questions_data)} questions, but {target_num_questions} were requested. API calls may have failed or returned fewer items than prompted.")
    else:
        print(f"Successfully generated and collected {len(final_questions_data)} questions.")

    return final_questions_data

def save_to_json(data, output_file_path):
    """Saves the data to a JSON file."""
    try:
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved questions to '{output_file_path}'")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    image_file_path = "image.jpg"  # <--- IMPORTANT: SET YOUR IMAGE PATH HERE OR PROVIDE AS ARGUMENT
    output_json_path = "output/image_questions.json"
    # Renamed for clarity in generate_image_questions_and_answers
    target_number_of_questions = 50 
    # --- End Configuration ---

    if not os.path.exists(image_file_path):
        print(f"Error: Image file not found at '{image_file_path}'. Please set the 'image_file_path' variable.")
    else:
        api_key = load_api_key()
        if api_key:
            output_dir = os.path.dirname(output_json_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")

            generated_data = generate_image_questions_and_answers(
                image_file_path,
                api_key,
                target_num_questions=target_number_of_questions # Pass the target number
            )
            if generated_data:
                save_to_json(generated_data, output_json_path)
            else:
                print("No questions were generated or an error occurred.")
        else:
            print("Could not proceed without an API key.")

    print("Script finished.")

# You'll need a function to encode the image if sending it as base64:
# import base64
# def encode_image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')
