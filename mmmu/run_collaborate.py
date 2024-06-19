import pdb
from datasets import load_dataset
from llm import *
from tqdm import tqdm
import json
from agent_prompt import *
import argparse


def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark


def get_last_processed_index(progress_file):
    """Retrieve the last processed index from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            last_index = f.read().strip()
            return int(last_index) if last_index else 0
    else:
        return 0


def update_progress(progress_file, index):
    """Update the last processed index in the progress file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(index))


def collaboration_func(ind, query, option, answer, model_name):
    i = 0
    answer_list = []
    description_ls = []
    while i < ind:
        flag = 0
        while True:
            prompt = meta_agent_prompt.format(query=query, option=option, answer=answer,
                                              description='\n-'.join(description_ls))
            description = llm_response(image, prompt, model_name)
            check_prompt = check_agent_prompt.format(query=query, option=option, description=description,
                                                     description_ls='\n-'.join(description_ls))
            check_result = llm_response(image, check_prompt)
            if 'discard' not in check_result.lower() or flag > 3:
                description_ls.append(description)
                break
            flag += 1

        prompt = multi_agent_prompt.format(query=query, option=option, description=description)
        sub_answer = llm_response(image, prompt, model_name)

        prompt = refine_agent_prompt.format(query=query, option=option, description=description,
                                            old_answer=answer, new_answer=sub_answer)
        new_answer = llm_response(image, prompt, model_name)

        answer_list.append({
            "time": i,
            "results": answer,
            "description": description,
            "sub_answer": sub_answer,
            "new_results": new_answer
        })

        answer = new_answer
        i = i + 1

    return answer_list, answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4v")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--ind", type=int, default=3)
    parser.add_argument("--end", type=int, default=31)
    args = parser.parse_args()
    model_name = args.model_name
    classes = [
        'Accounting',
        'Agriculture',
        'Architecture_and_Engineering',
        'Art_Theory',
        'Art',
        'Basic_Medical_Science', 'Biology',
        'Chemistry', 'Clinical_Medicine', 'Computer_Science',
        'Design', 'Diagnostics_and_Laboratory_Medicine',
        'Economics',
        'Electronics',
        'Energy_and_Power',
        'Finance',
        'Geography', 'History', 'Literature',
        'Manage',
        'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
        'Music',
        'Pharmacy',
        'Physics',
        'Psychology',
        'Public_Health',
        'Sociology',
    ]

    for cla in classes[args.start:args.end]:
        print(f"Class: {cla}")
        while True:
            try:
                sub_dataset_val = load_dataset('MMMU/MMMU', cla, split='validation', cache_dir="MMMU/dataset")
                break
            except:
                continue
        total_files = len(sub_dataset_val)
        print(f"Total Files: {total_files}")
        progress_file = f"output_collaborate/result_{model_name}_{cla}_collaborate.txt"
        start_index = get_last_processed_index(progress_file)
        dic = []
        ind = args.ind
        if os.path.exists(f"{os.path.join(progress_file.split('.')[0])}.jsonl"):
            already = read_jsonline(f"{os.path.join(progress_file.split('.')[0])}.jsonl")
            for data in already:
                dic.append(data["id"])

        with tqdm(total=total_files, desc="Processing files", initial=start_index) as pbar:
            for i in range(start_index, total_files):
                data = sub_dataset_val[i]

                if data["id"] in dic:
                    update_progress(progress_file, i + 1)
                    pbar.update(1)
                    continue

                image = data['image_1']
                question = data['question']
                print(data["question_type"])
                options = [f"{chr(65 + i)}. {item}" for i, item in enumerate(eval(data['options']))]
                options = '\n'.join(options)
                prompt = f'{question}\n{options}\nYou need to give reasons first and then give the answer with the format: "Answer:"'
                answer = llm_response(image, prompt, model_name)
                answer_list, answer = collaboration_func(ind, question, options, answer, model_name)
                answer_final = {"id": data["id"], "topic_difficulty": data["topic_difficulty"],
                                "subfield": data["subfield"],
                                "question": data['question'], "options": data['options'],
                                "golden_answer": data["answer"],
                                "question_type": data["question_type"], "answer": answer, "answer_list": answer_list}
                print(answer)
                with open(f"{os.path.join(progress_file.split('.')[0])}.jsonl", 'a+', encoding='utf-8') as f:
                    line = json.dumps(answer_final, ensure_ascii=False)
                    f.write(line + '\n')
                update_progress(progress_file, i + 1)
                pbar.update(1)
