from langchain.prompts import PromptTemplate

INSTRUCTION_META = """
{question}
This is your writing result:
{answer}

Now, you can create and collaborate with multiple experts to improve your writing result. Therefore, please describe in as much detail as possible the different skills and focuses you need from multiple experts individually. 
We will provide each expert with the same information and query. However, please note that each profession has its own specialization, so you can assign each expert to just one sub-task to ensure a more refined response. 
We will relay their responses to you in turn, allowing you to reorganize them into a better answer.
Please note that the description should be narrated in the second person, for example: You are a XXX.

These are the descriptions of the experts you have created before for this task:
{description}

Therefore, please remember you should not repeatedly create the same experts as described above.
Now, you can give the description for a new expert (Please note that only be one, do not give multiple at one time):
"""
meta_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "description"],
    template=INSTRUCTION_META,
)

INSTRUCTION_CHECK = """
{question}

We employ mulitple experts to answer this query. The following is a second-person introduction to the experts we have hired:
{description_ls}

Now, we will hire a new expert to help better respond to user query. Here is a second person description of the new expert:
{description}

Since hiring new experts takes extra time and money, please evaluate the new expert based on the following two criteria to decide whether they should be retained or not:
1. Based on the new expert's description, determine if they can effectively assist in answering users' questions.
2. The new experts are unique and do not overlap with previously hired experts.
The new expert must meet both of the above two criteria. If any of the criteria are not met, they should be discarded.
Give the reason first and then give the choice. If retaining, please reply with: 'Retain'. If discarding, please reply with: 'Discard'."
"""

check_agent_prompt = PromptTemplate(
    input_variables=["question", "description_ls", "description"],
    template=INSTRUCTION_CHECK,
)

INSTRUCTION_MULTI = """
{description}. 
{question}
"""

multi_agent_prompt = PromptTemplate(
    input_variables=["question", "description"],
    template=INSTRUCTION_MULTI,
)

INSTRUCTION_REFINE = """
{question}
This is your writing result:
{old_answer}

You invite an expert whose description is: \"{description}\"
This expert also give his answer based on his own professional knowledge: {new_answer}.

Now you can refine your writing result with his answer to better answer the question. 
Keep in mind that his answer may not be correct, so critically decide whether to accept his response or stick with your original one.
Revised Answer:
"""

refine_agent_prompt = PromptTemplate(
    input_variables=["question", "old_answer", "description", "new_answer"],
    template=INSTRUCTION_REFINE,
)

INSTRUCTION_SELF_REFINE = """

{question}
This is your writing result:
{answer}

There is the suggestion from an assistant:
Suggestion: {feedback}

Now you can refine your writing result with his suggestion to better answer the question. 
Keep in mind that his suggestion may not be correct, so critically decide whether to accept his response or stick with your original one.
Revised Answer:
"""

self_refine_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "feedback"],
    template=INSTRUCTION_SELF_REFINE,
)

INSTRUCTION_FEEDBACK = """
You are a helpful assistant that provides feedback on answers of coding.


{question}
This is the answer from a student: {answer}.

Please do not refine the answer but give some insightful suggestions for the student to help him better answer the question.
Suggestion:
"""

feedback_agent_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=INSTRUCTION_FEEDBACK,
)

PERSONA_GEN = """
When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary. 
Here are some examples: 
--
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once. 
Input: 6 12 1 1 
Participants except you: Math Expert

--
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid. 
Participants except you: Poet; Computer Scientist; Ten year old child

--
Please remember you should give the participants directly and these participants are separated by semicolons (;)
Task: 
{question}
Participants except you: 
"""

persona_gen_agent_prompt = PromptTemplate(
    input_variables=["question"],
    template=PERSONA_GEN,
)

SUGGEST_GEN = """
You are a {persona}.


{question}

Do not answer the query but give some suggestions for an AI assistant for better answer the question.
Suggestion:
"""

suggest_gen_agent_prompt = PromptTemplate(
    input_variables=["question", "persona"],
    template=SUGGEST_GEN,
)

SPP_INSTRUCTION = """

{question}

These are suggestion from others:
{suggestion}

Now you can refine your writing result with his answer to better answer the question. You need to give reasons first and then give the answer with the format: \"Final Answer: choice: XX\"
Answer:
"""

spp_agent_prompt = PromptTemplate(
    input_variables=["query", "option", "suggestion"],
    template=SPP_INSTRUCTION,
)

SPP_INSTRUCTION_FEEDBACK = """
You are a {persona}

{question}
This is the answer from a student: {answer}.

Please do not refine the answer but give some insightful suggestions for the student to help him better answer the question. If you think the answer is good and no suggestion, please only output \"Well Done!\"
Suggestion:
"""

spp_feedback_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "persona"],
    template=SPP_INSTRUCTION_FEEDBACK,
)

SPP_INSTRUCTION_SELF_REFINE = """
{question}

This is your writing result:
Answer: {answer}.

There are suggestions from others:
{suggestion}

Now you can refine your writing result with his answer to better answer the question. You need to give reasons first and then give the answer with the format: \"Final Answer: choice: XX\"
Revised Answer:
"""

spp_self_refine_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "suggestion"],
    template=SPP_INSTRUCTION_SELF_REFINE,
)
