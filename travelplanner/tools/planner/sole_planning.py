import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt, \
#    react_reflect_planner_agent_prompt, reflect_prompt
from agents.prompts import *
from agents.persona_prompt import *
# from utils.func import get_valid_name_city,extract_before_parenthesis, extract_numbers_from_filenames
import json
import time
from langchain.callbacks import get_openai_callback
import pdb
from tqdm import tqdm
from tools.planner.apis import *
import openai
import argparse
from datasets import load_dataset
import logging

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data


def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)

def collaboration_func(ind, reference_information, query, planner_results):
    
    i = 0
    planner_list = []
    description_ls = []
    while i < ind:
        flag = 0
        while True:
            meta_planner = Meta_Planner(model_name=args.model_name, agent_prompt=meta_planner_agent_prompt)
            description = meta_planner.run(reference_information, query, planner_results, '\n-'.join(description_ls))
            check_result = "a"
            if 'discard' not in check_result.lower() or flag > 3:
                description_ls.append(description)
                break
            flag += 1
        
        multi_planner = Multi_Planner(model_name=args.model_name, agent_prompt=multi_planner_agent_prompt)
        sub_answer = multi_planner.run(reference_information, query, description)

        refine_planner = Refine_Planner(model_name=args.model_name, agent_prompt=refine_planner_agent_prompt)
        new_planner_results = refine_planner.run(reference_information, query, planner_results, description, sub_answer)

        planner_list.append({
            "time": i,
            "planner_results":planner_results,
            "description": description,
            "sub_answer": sub_answer,
            "new_planner_results": new_planner_results
                             })

        planner_results = new_planner_results

        i = i+1

    return planner_list, planner_results

def group_func(ind, group_num, select_strategy, reference_information, query, planner_results):
    
    i = 0
    planner_list = []
    description_ls = []
    while i < ind:
        
        group_answers = []
        for j in range(group_num):
            meta_planner = Meta_Planner(model_name=args.model_name, agent_prompt=meta_planner_agent_prompt)
            description = meta_planner.run(reference_information, query, planner_results, '\n-'.join(description_ls))
            if description.strip() in description_ls:
                print("Same Expert. Discard!")
                continue
            check_planner = Check_Planner(model_name=args.model_name, agent_prompt=check_planner_agent_prompt)
            check_result = check_planner.run(reference_information, query, description_ls, description)
            print(check_result)
            if 'discard' not in check_result.lower():
                description_ls.append(description.strip())
            else:
                continue
            multi_planner = Multi_Planner(model_name=args.model_name, agent_prompt=multi_planner_agent_prompt)
            sub_answer = multi_planner.run(reference_information, query, description)
            group_answers.append([description, sub_answer])
        if len(group_answers) == 0:
            break
        else:
            print(f"Group Answers: {len(group_answers)}")
        import random
        random.seed(123)
        if select_strategy == 'random':
            random.shuffle(group_answers)
            description = group_answers[0][0]
            sub_answer = group_answers[0][1]
            refine_planner = Refine_Planner(model_name=args.model_name, agent_prompt=refine_planner_agent_prompt)
            new_planner_results = refine_planner.run(reference_information, query, planner_results, description, sub_answer)
        elif select_strategy == 'pk':
            pk_planner = PK_Planner(model_name=args.model_name, agent_prompt=pk_planner_agent_prompt)
            group_answers_prompt = ''
            num = 0
            for group_answer in group_answers:
                group_answers_prompt += f"\nExpert #{num} Description: {group_answer[0]}\nTravel Plan{group_answer[1]}\n"
                num = num+1
            pk_answer = pk_planner.run(reference_information, query, len(group_answers), group_answers_prompt)
            try:
                expert_num = int(pk_answer.split("#")[-1].replace(".","").strip()[0])
            except:
                temp = pk_answer.split("Final Answer:")[-1]
                expert_num = 0
                for gn in range(group_num):
                    if str(gn) in temp:
                        expert_num = gn
                        break
            description = group_answers[expert_num][0]
            sub_answer = group_answers[expert_num][1]
            refine_planner = Refine_Planner(model_name=args.model_name, agent_prompt=refine_planner_agent_prompt)
            new_planner_results = refine_planner.run(reference_information, query, planner_results, description, sub_answer)
        elif select_strategy == 'all':
            all_planner = All_Planner(model_name=args.model_name, agent_prompt=all_planner_agent_prompt)
            sub_answer = ""
            group_answers_prompt = ''
            num = 0
            for group_answer in group_answers:
                group_answers_prompt += f"\nExpert {num} Description: {group_answer[0]}\nTravel Plan{group_answer[1]}\n"
                num = num+1
            new_planner_results = all_planner.run(reference_information, query, planner_results, group_answers_prompt, len(group_answers))

        

        planner_list.append({
            "time": i,
            "planner_results":planner_results,
            "description": description,
            "group_answers": group_answers,
            "sub_answer": sub_answer,
            "new_planner_results": new_planner_results
                             })

        planner_results = new_planner_results

        i = i+1

    return planner_list, planner_results





def refine_func(ind, reference_information, query, planner_results):
    
    i = 0
    planner_list = []
    while i < ind:
        
        feedback_planner = Feedback_Planner(model_name=args.model_name, agent_prompt=feedback_planner_agent_prompt)
        feedback_description = feedback_planner.run(reference_information, query, planner_results)

        self_refine_planner = Self_Refine_Planner(model_name=args.model_name, agent_prompt=self_refine_planner_agent_prompt)
        new_planner_results = self_refine_planner.run(reference_information, query, planner_results, feedback_description)

        planner_list.append({
            "time": i,
            "planner_results":planner_results,
            "refine": feedback_description,
            "new_planner_results": new_planner_results
                             })
        #pdb.set_trace()
        planner_results = new_planner_results
        i = i+1

    return planner_list, planner_results


def overgen_func(ind, reference_information, query):
    
    i = 0
    planner_list = []
    overgen_planner = Overgen_Planner(model_name=args.model_name, agent_prompt=overgen_planner_agent_prompt)
    overgeneration = overgen_planner.run(reference_information, query)

    select_planner = Select_Planner(model_name=args.model_name, agent_prompt=select_planner_agent_prompt)
    planner_results = select_planner.run(reference_information, query, overgeneration)
    planner_list.append({
        "planner_results": planner_results,
        "overgeneration": overgeneration
    })

    return planner_list, planner_results


def ssp_func(ind, reference_information, query):
    persona_gen = Persona_Generator(model_name=args.model_name, agent_prompt=persona_gen_agent_prompt)
    persona_ls = persona_gen.run(query)
    persona_ls = persona_ls.split(";")
    suggest_inital = []
    suggest_gen = Suggestion_Generator(model_name=args.model_name, agent_prompt=suggest_gen_agent_prompt)
    for persona in persona_ls:
        suggest = suggest_gen.run(reference_information, query, persona)
        suggest_inital.append(f"{persona.strip()}: {suggest.strip()}")

    ssp_planner = SSP_Planner(model_name=args.model_name, agent_prompt=ssp_planner_agent_prompt)
    planner_results = ssp_planner.run(reference_information, query, '\n'.join(suggest_inital))
    planner_list = []
    planner_list.append(
        {
            "persona": persona_ls,
            "suggest_inital": suggest_inital,
            "planner_results":planner_results,
        }
    )
    i = 0
    while i < ind:
        suggest_temp = []
        for persona in persona_ls:
            ssp_feedback_planner = SSP_Feedback_Planner(model_name=args.model_name, agent_prompt=ssp_feedback_planner_agent_prompt)
            ssp_feedback_description = ssp_feedback_planner.run(reference_information, query, planner_results, persona)
            if ssp_feedback_description == "Well Done!":
                continue
            else:
                suggest_temp.append(f"{persona.strip()}: {ssp_feedback_description.strip()}")
        if len(suggest_temp) == 0:
            break
        ssp_self_refine_planner = SSP_Self_Refine_Planner(model_name=args.model_name, agent_prompt=ssp_self_refine_planner_agent_prompt)
        new_planner_results = ssp_self_refine_planner.run(reference_information, query, planner_results, '\n'.join(suggest_temp))

        planner_list.append({
            "time": i,
            "planner_results":planner_results,
            "refine": suggest_temp,
            "new_planner_results": new_planner_results
                             })
        #pdb.set_trace()
        planner_results = new_planner_results
        i = i+1

    return planner_list, planner_results

def prompt_refine_func(ind, reference_information, query, planner_results):
    
    i = 0
    planner_list = []
    description_ls = []
    while i < ind:
        
        promptrefine_planner = PromptRefine_Planner(model_name=args.model_name, agent_prompt=promptrefine_planner_agent_prompt)
        feedback_description = promptrefine_planner.run(reference_information, query, planner_results)

        multi_planner = Multi_Planner(model_name=args.model_name, agent_prompt=multi_planner_agent_prompt)
        new_planner_results = multi_planner.run(reference_information, query, feedback_description)

        planner_list.append({
            "time": i,
            "planner_results":planner_results,
            "refine": feedback_description,
            "new_planner_results": new_planner_results
                             })
        #pdb.set_trace()
        planner_results = new_planner_results
        i = i+1

    return planner_list, planner_results


def suggest_func(ind, reference_information, query, planner_results):
    
    i = 0
    planner_list = []
    description_ls = []
    while i < ind:
        meta_planner = Meta_Planner(model_name=args.model_name, agent_prompt=meta_planner_agent_prompt)
        description = meta_planner.run(reference_information, query, planner_results, '\n-'.join(description_ls))

        suggest_planner = Suggest_Planner(model_name=args.model_name, agent_prompt=suggest_planner_agent_prompt)
        suggestion = suggest_planner.run(reference_information, query, planner_results, description)

        self_refine_planner = Self_Refine_Planner(model_name=args.model_name, agent_prompt=self_refine_planner_agent_prompt)
        new_planner_results = self_refine_planner.run(reference_information, query, planner_results, suggestion)

        planner_list.append({
            "time": i,
            "planner_results":planner_results,
            "description": description,
            "suggestion": suggestion,
            "new_planner_results": new_planner_results
                             })
        #pdb.set_trace()
        planner_results = new_planner_results
        i = i+1

    return planner_list, planner_results
    
def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--rewrite", type=bool, default=False)
    parser.add_argument("--ind", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=180)
    parser.add_argument("--group_num", type=int, default=2)
    parser.add_argument("--select_strategy", type=str, default="random")

    args = parser.parse_args()
    directory = f'{args.output_dir}/{args.set_type}'
    ind = args.ind
    if args.set_type == 'validation':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    elif args.set_type == 'test':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'test')['test']
    numbers = [i for i in range(1, len(query_data_list) + 1)]
    #
    if args.strategy == 'direct':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'cot':
        planner = Planner(model_name=args.model_name, agent_prompt=cot_planner_agent_prompt)
    elif args.strategy == 'react':
        planner = ReactPlanner(model_name=args.model_name, agent_prompt=react_planner_agent_prompt)
    elif args.strategy == 'reflexion':
        planner = ReactReflectPlanner(model_name=args.model_name, agent_prompt=react_reflect_planner_agent_prompt,
                                      reflect_prompt=reflect_prompt)
    elif args.strategy == 'collaboration':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'refine':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'group':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'suggest':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'promptrefine':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)

    with get_openai_callback() as cb:
        for number in tqdm(numbers[args.start:args.end]):
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
                result = [{}]
            else:
                result = json.load(
                    open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))
            
            if args.rewrite == False:
                if args.strategy in ['collaboration', 'refine', 'ssp', 'overgen', 'suggest', 'promptrefine']:
                    if f'{args.model_name}_{args.strategy}_{ind}_sole-planning_results' in result[-1]:
                        print(f"Do not write: {args.output_dir}/{args.set_type}/generated_plan_{number}.json")
                        #import pdb
                        #pdb.set_trace()
                        continue
                elif args.strategy in ['group']:
                    if f'{args.model_name}_{args.strategy}_{args.ind}_{args.group_num}_{args.select_strategy}_sole-planning_results' in result[-1]:
                        print(f"Do not write: {args.output_dir}/{args.set_type}/generated_plan_{number}.json")
                        #import pdb
                        #pdb.set_trace()
                        continue
                else:
                    if f'{args.model_name}_{args.strategy}_sole-planning_results' in result[-1]:
                        print(f"Do not write: {args.output_dir}/{args.set_type}/generated_plan_{number}.json")
                        continue

            
            query_data = query_data_list[number - 1]
            reference_information = query_data['reference_information']
            #import pdb
            #pdb.set_trace()
            while True:
                if args.strategy in ['react', 'reflexion']:
                    planner_results, scratchpad = planner.run(reference_information, query_data['query'])
                elif args.strategy == 'collaboration':
                    ind = args.ind
                    refine_plan = planner.run(reference_information, query_data['query'])
                    planner_list, planner_results = collaboration_func(ind, reference_information, query_data['query'], refine_plan)
                elif args.strategy == 'group':
                    ind = args.ind
                    group_num = args.group_num
                    select_strategy = args.select_strategy
                    refine_plan = planner.run(reference_information, query_data['query'])
                    planner_list, planner_results = group_func(ind, group_num, select_strategy, reference_information, query_data['query'], refine_plan)
                elif args.strategy == 'refine':
                    ind = args.ind
                    refine_plan = planner.run(reference_information, query_data['query'])
                    planner_list, planner_results = refine_func(ind, reference_information, query_data['query'], refine_plan)
                elif args.strategy == 'promptrefine':
                    ind = args.ind
                    refine_plan = planner.run(reference_information, query_data['query'])
                    planner_list, planner_results = prompt_refine_func(ind, reference_information, query_data['query'], refine_plan)
                elif args.strategy == 'suggest':
                    ind = args.ind
                    refine_plan = planner.run(reference_information, query_data['query'])
                    planner_list, planner_results = suggest_func(ind, reference_information, query_data['query'], refine_plan)
                elif args.strategy == 'ssp':
                    ind = args.ind
                    planner_list, planner_results = ssp_func(ind, reference_information, query_data['query'])
                elif args.strategy == 'overgen':
                    ind = args.ind
                    planner_list, planner_results = overgen_func(ind, reference_information, query_data['query'])
                else:
                    ind = args.ind
                    planner_results = planner.run(reference_information, query_data['query'])
                if planner_results != None:
                    break
            print(planner_results)
            if args.strategy in ['react', 'reflexion']:
                result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad
            result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results'] = planner_results
            if args.strategy in ['collaboration', 'refine', 'ssp', 'overgen', 'suggest', 'promptrefine']:
                result[-1][f'{args.model_name}_{args.strategy}_sole-planning_multi'] = planner_list
                result[-1][f'{args.model_name}_{args.strategy}_{ind}_sole-planning_results'] = planner_results
            if args.strategy in ['group']:
                result[-1][f'{args.model_name}_{args.strategy}_sole-planning_multi'] = planner_list
                result[-1][f'{args.model_name}_{args.strategy}_{ind}_{group_num}_{select_strategy}_sole-planning_results'] = planner_results
            print("write to json file")
            with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
                json.dump(result, f, indent=4)
            #pdb.set_trace()
        print(cb)
