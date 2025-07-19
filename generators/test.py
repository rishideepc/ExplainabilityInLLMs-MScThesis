# # from datasets import load_dataset

# # # # dataset = load_dataset("truthful_qa", "generation")
# # # # questions = dataset["validation"]["question"][:3]

# # # # Load initial experiment questions - CommonsenseQA
# # # dataset = load_dataset("wics/strategy-qa")
# # # questions = [entry["question"] for entry in dataset["validation"]]
# # # # questions = questions[:4]

# # # Load initial experiment questions - MedQA
# # dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
# # questions = [entry["question"] for entry in dataset["validation"]]
# # # questions = questions[:2290]

# # print(len(questions))



# # # import psutil
# # # import GPUtil

# # # # RAM
# # # ram = psutil.virtual_memory()
# # # print(f"Total RAM: {ram.total / 2e9:.2f} GB")

# # # # VRAM
# # # gpus = GPUtil.getGPUs()
# # # for gpu in gpus:
# # #     print(f"GPU: {gpu.name}")
# # #     print(f"  Total VRAM: {gpu.memoryTotal} MB")



# from datasets import load_dataset
# import pandas as pd

# # Load MIMIC-NLE dataset
# dataset = load_dataset("Fraser/MIMIC-NLE")

# # Print available splits
# print("Available splits:", dataset.keys())

# # Peek into the first few entries of the 'train' split
# train_data = dataset["train"]
# print(f"Total entries in train split: {len(train_data)}")

# # Convert to pandas DataFrame for easier inspection
# df = pd.DataFrame(train_data[:5])  # Change 5 to any number of samples you want to inspect

# # Display available columns
# print("Columns in dataset:\n", df.columns)

# # Display sample rows
# print("\nSample data entries:\n", df.T)



# from mimic_nle.extract_mimic_nle import extract_sentences
# from pathlib import Path

# # Point to your local MIMIC‑CXR report dump
# reports_path = "c:\\users\\rishi\\appdata\\local\\temp\\pip-req-build-oyvgsi99"
# reports_dir = Path(reports_path)

# # Run the extraction
# nle_datasets = extract_sentences(str(reports_dir))

# # This returns a dict: {'train': [...], 'dev': [...], 'validation': [...]}
# print({split: len(nle_datasets[split]) for split in ['train','dev','validation']})
# # e.g., {'train': 25000, 'dev': 5000, 'validation': 5500}

# # Inspect one example
# sample = nle_datasets['validation'][3]
# print("Report ID:", sample["report_ID"])
# print("Sentence:", sample["sentence"])
# print("Label(s):", sample.get("pathology_labels"))
# print("Natural‑language explanation:", sample["nle"])


# from datasets import load_dataset

# # dataset = load_dataset("truthful_qa", "generation")
# dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
# # dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
# questions = [entry["question"] for entry in dataset["validation"]]
# question = questions[2]
# print("Question: ", question, "\n")

# answers = [entry["answer"] for entry in dataset["validation"]]
# answer = answers[2]
# print("Answer: ", answer, "\n")

# optionss = [entry["options"] for entry in dataset["validation"]]
# options = optionss[2]
# print("Options: ", options, "\n")

# meta_infos = [entry["meta_info"] for entry in dataset["validation"]]
# meta_info = meta_infos[2]
# print("Meta_info: ", meta_info, "\n")

# answer_idxs = [entry["answer_idx"] for entry in dataset["validation"]]
# answer_idx = answer_idxs[2]
# print("Answer_idx: ", answer_idx, "\n")

# metamap_phrasess = [entry["metamap_phrases"] for entry in dataset["validation"]]
# metamap_phrases = metamap_phrasess[2]
# print("Metamap_phrases: ", metamap_phrases, "\n")





# dataset = load_dataset("truthful_qa", "generation")
# question= dataset["validation"]["type"][3]
# category = dataset["validation"]["category"][3]
# question = dataset["validation"]["question"][3]  # adjust sample size here
# best_answer = dataset["validation"]["best_answer"][3]
# correct_answers = dataset["validation"]["correct_answers"][3]
# incorrect_answers = dataset["validation"]["incorrect_answers"][3]
# source = dataset["validation"]["source"][3]
# mc_2 = dataset["validation"]["mc2_targets"][3]
# print(f"Number of entries in training split: {len(dataset)}\n")
# print("The dataset: -\n")
# print(dataset)

# print(type_)
# print(category)
# print(question)
# print(best_answer)
# print(correct_answers)
# print(len(dataset["validation"]["correct_answers"][3]))
# print(incorrect_answers)
# print(len(dataset["validation"]["incorrect_answers"][3]))
# print(source)



# sample = dataset[3]
# for key, value in sample.items():
#     print(f"{key}:\n{value}\n{'-'*60}")

# for item in dataset[3].items(): 
#     print(item)


################################################################################################################

# from datasets import load_dataset
# import random

# # Step 2: Load the TruthfulQA dataset
# dataset = load_dataset("truthful_qa", "generation")

# # Step 2: Initialize variables
# true_claims = []
# false_claims = []
# desired_true = 409
# desired_false = 408

# # Step 3: Shuffle the dataset to randomize selection order
# data_items = list(dataset["validation"])
# random.shuffle(data_items)

# # Step 4: Generate claims with balanced labels
# for item in data_items:
#     correct = item["correct_answers"]
#     incorrect = item["incorrect_answers"]

#     if len(true_claims) < desired_true and correct:
#         claim = random.choice(correct)
#         true_claims.append({
#             "claim": claim,
#             "label": "true",
#             "question": item["question"],
#             "source": item["source"]
#         })
#     elif len(false_claims) < desired_false and incorrect:
#         claim = random.choice(incorrect)
#         false_claims.append({
#             "claim": claim,
#             "label": "false",
#             "question": item["question"],
#             "source": item["source"]
#         })

#     # Stop once both lists are full
#     if len(true_claims) >= desired_true and len(false_claims) >= desired_false:
#         break

    

# # Step 5: Combine and shuffle the final dataset
# final_dataset = true_claims + false_claims
# random.shuffle(final_dataset)

# for idx, item in enumerate(final_dataset, start=2):
#     item["id"] = idx

# # Step 6: (Optional) Print a few examples
# for i in range(827):
#     print(f"{i+2}. [{final_dataset[i]['label']}] {final_dataset[i]['claim']}")

# # Step 7: (Optional) Save to JSON or CSV
# import json
# with open("truthful_claims_dataset.json", "w") as f:
#     json.dump(final_dataset, f, indent=2)


############################################################################################################
# import random
# from datasets import load_dataset
# import json

# # Step 2: Load MedQA dataset (USMLE, 4-option MCQ format)
# dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
# data = list(dataset["validation"])  # 2273 entries

# # Step 2: Set number of true/false claims
# num_true_claims = 637
# num_false_claims = 636

# # Step 3: Shuffle dataset to randomize assignment
# random.shuffle(data)

# # Step 4: Prepare true and false claim source sets
# true_claims = []
# false_claims = []

# for i, item in enumerate(data):
#     question = item["question"]
#     options = item["options"]
#     answer_idx = item["answer_idx"]
#     correct_answer = options[answer_idx]

#     # --- TRUE claim setup ---
#     if len(true_claims) < num_true_claims:
#         true_claims.append({
#             "question": question,
#             "option": correct_answer,
#             "label": "true",
#             "source_idx": i
#         })

#     # --- FALSE claim setup ---
#     elif len(false_claims) < num_false_claims:
#         incorrect_indices = [idx for idx in options if idx != answer_idx]
#         incorrect_option_keys = [k for k in options.keys() if k != answer_idx]
#         random_false_key = random.choice(incorrect_option_keys)
#         false_answer = options[random_false_key]
#         false_claims.append({
#             "question": question,
#             "option": false_answer,
#             "label": "false",
#             "source_idx": i
#         })

#     # Exit loop if both lists are filled
#     if len(true_claims) == num_true_claims and len(false_claims) == num_false_claims:
#         break

# # Step 5: Combine and shuffle
# final_dataset = true_claims + false_claims
# random.shuffle(final_dataset)

# # Step 6: Add serial numbers
# for i, item in enumerate(final_dataset):
#     item["id"] = i + 2

# # Step 7: Preview (optional)
# for i in range(5):
#     print(f"{final_dataset[i]['id']}. [{final_dataset[i]['label']}] Q: {final_dataset[i]['question']}\n   Option: {final_dataset[i]['option']}\n")

# # Step 8: Save to JSON (ready for LLM processing)
# with open("medclaim_prompt_inputs.json", "w") as f:
#     json.dump(final_dataset, f, indent=2)

# print(f"\n✅ MedClaim prompt dataset generated with {len(final_dataset)} items (637 true, 636 false).")



########################################################################################################################


# import sys
# import os
# project_root = os.path.abspath('..')
# sys.path.append(project_root)

# import json
# import subprocess
# import time

# INPUT_FILE = "generators\medclaim_prompt_inputs.json"
# OUTPUT_FILE = "medclaim_final.json"
# MODEL_NAME = "mistral"  # Ensure this is available in Ollama (use `ollama list` to confirm)

# # Load data
# with open(INPUT_FILE, "r") as f:
#     dataset = json.load(f)

# generated_dataset = []

# # Prompt template
# def create_prompt(question, option):
#     return f"""You are a medical expert.

# Based on the following medical question and a multiple-choice answer option, generate a single complete medical claim. Don't worry about the claim being true or false, just generate a grammatically correct claim sentence using the question and the answer option. Every claim should be self-contained/independent and should not assume that the reader has context about the scenario. Avoid beginning with phrases such as "Given the..." or "Based on the..."

# Question: {question}

# Option: {option}

# Claim:



# Here is a one-shot example: -

# Question: A post-mortem lung examination of a 68-year-old male overweight male with evidence of chronic lower extremity edema, a 60 pack-year smoking history and daily productive cough would be most likely to reveal:
# Option: Evidence of a necrotizing infection
# Claim: The post-mortem lung examination of a 68-year-old overweight male with chronic lower extremity edema, a 60 pack-year smoking history and daily productive cough is most likely to reveal evidence of a necrotizing infection.
# """

# # Function to call Ollama with Qwen model
# def query_qwen(prompt, retries=3, wait_time=3):

#     for attempt in range(retries):
#         try:
#             result = subprocess.run(
#                 ["ollama", "run", MODEL_NAME],
#                 input=prompt.encode("utf-8"),
#                 capture_output=True,
#                 check=True
#             )
#             output = result.stdout.decode("utf-8").strip()
#             return output
#         except subprocess.CalledProcessError as e:
#             print(f"⚠️ Error on attempt {attempt + 2}: {e}")
#             time.sleep(wait_time)
#     return "ERROR: Failed to generate claim"

# # Generate claims
# for i, item in enumerate(dataset):
#     print(f"🧠 [{i+2}/{len(dataset)}] Generating claim for ID {item['id']} ({item['label']})")
#     prompt = create_prompt(item["question"], item["option"])
#     claim = query_qwen(prompt)

#     # Store result
#     generated_dataset.append({
#         "id": item["id"],
#         "question": item["question"],
#         "option": item["option"],
#         "claim": claim,
#         "label": item["label"],
#         "source_idx": item["source_idx"]
#     })

#     # Optional: save progress checkpoint
#     if (i + 2) % 50 == 2:
#         with open("medclaim_progress.json", "w") as f:
#             json.dump(generated_dataset, f, indent=2)
#         print(f"💾 Progress saved at {i+2} items.")

# # Final save
# with open(OUTPUT_FILE, "w") as f:
#     json.dump(generated_dataset, f, indent=2)

# print(f"\n✅ All done! Final dataset saved to '{OUTPUT_FILE}' with {len(generated_dataset)} entries.")


# ##############################################################################################################################



# from datasets import load_dataset

# dataset = load_dataset("wics/strategy-qa")
# print(dataset)

# qids = [entry["qid"] for entry in dataset["validation"]]
# print("QID: ", qids[2])

# terms = [entry["term"] for entry in dataset["validation"]]
# print("Term: ", terms[2])

# descriptions = [entry["description"] for entry in dataset["validation"]]
# print("Description: ", descriptions[2])

# questions = [entry["question"] for entry in dataset["validation"]]
# print("Question: ", questions[2])

# answers = [entry["answer"] for entry in dataset["validation"]]
# print("Answer: ", answers[2])

# factss = [entry["facts"] for entry in dataset["validation"]]
# print("Facts: ", factss[2])

# decompositions = [entry["decomposition"] for entry in dataset["validation"]]
# print("Decomposition: ", decompositions[2])


# c, k = 2, 2
# for i in range(2, len(answers)):
#     if answers[i]==False:
#         c+=2

#     elif answers[i]==True:
#         k+=2

# print("false: ", c, "\n", "true: ", k)
# print(c+k)


#################################################################################################################################


# import sys
# import os
# project_root = os.path.abspath('..')
# sys.path.append(project_root)

# import json
# import subprocess
# from datasets import load_dataset
# import time

# # INPUT_FILE = "generators\medclaim_prompt_inputs.json"
# OUTPUT_FILE = "strategyclaim_final.json"
# MODEL_NAME = "mistral"  # Ensure this is available in Ollama (use `ollama list` to confirm)

# dataset = load_dataset("wics/strategy-qa")

# questions = [entry["question"] for entry in dataset["validation"]]

# answers = [entry["answer"] for entry in dataset["validation"]]

# qids = [entry["qid"] for entry in dataset["validation"]]

# # Load data
# # with open(INPUT_FILE, "r") as f:
# #     dataset = json.load(f)

# generated_dataset = []

# # Prompt template
# def create_prompt(question, answer):
#     return f"""
    
# Few shot examples: -

# Question:  Are more people today related to Genghis Khan than Julius Caesar?
# Answer:  True

# Converted Claim: More people today are related to Genghis Khan than Julius Caesar



# Question:  Could the members of The Police perform lawful arrests?
# Answer:  False

# Converted Claim: The members of The Police cannot perform lawful arrests.



# Now similar to the above examples, convert the following Question-Answer pair into a claim statement: -

# Question:  {question}
# Answer:  {answer}

# Just generate the claim statement as a response to this prompt, DO NOT generate any associated keys such as "Claim Statement:" or "Claim:" or "The claim is:" or "Converted Claim:"
# """

# # Function to call Ollama with Qwen model
# def query_qwen(prompt, retries=3, wait_time=3):

#     for attempt in range(retries):
#         try:
#             result = subprocess.run(
#                 ["ollama", "run", MODEL_NAME],
#                 input=prompt.encode("utf-8"),
#                 capture_output=True,
#                 check=True
#             )
#             output = result.stdout.decode("utf-8").strip()
#             return output
#         except subprocess.CalledProcessError as e:
#             print(f"⚠️ Error on attempt {attempt + 2}: {e}")
#             time.sleep(wait_time)
#     return "ERROR: Failed to generate claim"

# # Generate claims
# for i in range(len(answers)):
#     print(f"🧠 [{i+2}/{len(answers)}] Generating claim for ID {qids[i]} ({answers[i]})")
#     prompt = create_prompt(questions[i], answers[i])
#     claim = query_qwen(prompt)

#     # Store result
#     generated_dataset.append({
#         "qid": qids[i],
#         "question": questions[i],
#         "claim": claim,
#         "label/answer": answers[i]
#     })

#     # Optional: save progress checkpoint
#     if (i + 2) % 50 == 2:
#         with open("medclaim_progress.json", "w") as f:
#             json.dump(generated_dataset, f, indent=2)
#         print(f"💾 Progress saved at {i+2} items.")

# # Final save
# with open(OUTPUT_FILE, "w") as f:
#     json.dump(generated_dataset, f, indent=2)

# print(f"\n✅ All done! Final dataset saved to '{OUTPUT_FILE}' with {len(generated_dataset)} entries.")
#########################################################################################################################

# from datasets import load_dataset


# dataset = load_dataset("commonsense_qa")
# # print(dataset)
# # questions = [entry["question"] for entry in dataset["validation"]]
# # questions = questions[:1221]

# # dataset = load_dataset("wics/strategy-qa")
# # print(dataset)

# ids = [entry["id"] for entry in dataset["validation"]]
# print("ID: ", ids[2])

# questions = [entry["question"] for entry in dataset["validation"]]
# print("Question: ", questions[2])

# question_concepts = [entry["question_concept"] for entry in dataset["validation"]]
# print("Question_Concept: ", question_concepts[2])

# choicess = [entry["choices"] for entry in dataset["validation"]]
# print("Choices: ", choicess[2])

# answerKeys = [entry["answerKey"] for entry in dataset["validation"]]
# print("Answer Key: ", answerKeys[2])


# c, k = 2, 2
# for i in range(2, len(answers)):
#     if answers[i]==False:
#         c+=2

#     elif answers[i]==True:
#         k+=2

# print("false: ", c, "\n", "true: ", k)
# print(c+k)

####################################################################################################################################


# import sys
# import os
# project_root = os.path.abspath('..')
# sys.path.append(project_root)

# import json
# import random
# import subprocess
# from datasets import load_dataset
# from tqdm import tqdm

# # Seed for reproducibility
# random.seed(42)

# # Load the CommonsenseQA validation split
# dataset = load_dataset("commonsense_qa")["validation"]

# # Helper to map answerKey (e.g., "A") to the corresponding text
# def get_option_text(entry, key):
#     idx = entry["choices"]["label"].index(key)
#     return entry["choices"]["text"][idx]

# # Helper to get a random incorrect answer
# def get_random_incorrect_option(entry, correct_key):
#     labels = entry["choices"]["label"]
#     texts = entry["choices"]["text"]
#     incorrect = [(l, t) for l, t in zip(labels, texts) if l != correct_key]
#     return random.choice(incorrect)[1]

# # LLM prompt function using Mistral via Ollama CLI
# def generate_claim_with_qwen(question, option):
#     prompt = f"""
# Few shot examples: -

# Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?
# Answer: bank
# Converted Claim: A revolving door is convenient for two direction travel, but it also serves as a security measure at a bank.


# Question: What do people aim to do at work?
# Answer: complete job
# Converted Claim: People aim to complete the job at work.



# Now similar to the above examples, convert the following Question-Answer pair into a claim statement: -

# Question: {question}
# Answer: {option}

# Just generate the claim statement as a response to this prompt, DO NOT generate any associated keys such as "Claim Statement:" or "Claim:" or "The claim is:" or "Converted Claim:"
# """
#     result = subprocess.run(
#         ["ollama", "run", "mistral"],
#         input=prompt.encode("utf-8"),
#         capture_output=True
#     )
#     response = result.stdout.decode("utf-8").strip()
#     # Strip prompt echoes and artifacts if present
#     return response.split("Claim:")[-1].strip()

# # Build dataset entries
# output_data = []

# print("\n🟢 Generating TRUE claims (correct answers)...")
# for i in tqdm(range(611)):
#     entry = dataset[i]
#     question = entry["question"]
#     correct_key = entry["answerKey"]
#     option = get_option_text(entry, correct_key)
#     claim = generate_claim_with_qwen(question, option)
#     output_data.append({
#         "id": i + 1,
#         "question": question,
#         "option": option,
#         "claim": claim,
#         "label": "true",
#         "source_id": entry["id"]
#     })

# print("\n🔴 Generating FALSE claims (random wrong answers)...")
# for i in tqdm(range(611, 1221)):
#     entry = dataset[i]
#     question = entry["question"]
#     correct_key = entry["answerKey"]
#     option = get_random_incorrect_option(entry, correct_key)
#     claim = generate_claim_with_qwen(question, option)
#     output_data.append({
#         "id": i + 1,
#         "question": question,
#         "option": option,
#         "claim": claim,
#         "label": "false",
#         "source_id": entry["id"]
#     })

# # Shuffle the final dataset
# print("\n🔀 Shuffling final dataset...")
# random.shuffle(output_data)

# # Save to JSON
# output_path = "commonsenseclaim.json"
# with open(output_path, "w") as f:
#     json.dump(output_data, f, indent=2)

# print(f"\n✅ CommonsenseClaim dataset saved to '{output_path}' with {len(output_data)} entries.")

##################################################################################################################################

# from datasets import load_dataset

# dataset = load_dataset("truthful_claim")["test"]
# sample = dataset[0]  # Subset for testing
# print(f"Sample: {sample}")
# question = sample["claim"]
# answer = sample["label"].upper()


# print(f"Question: {question}")
# print(f"Answer: {answer}")



# Left for regenration (as on 19.7.25): StrategyClaim - Qwen; MedClaim - Qwen