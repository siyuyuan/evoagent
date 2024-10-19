from langchain.prompts import PromptTemplate

SPY_INSTRUCTION_META = """
{question}
This is your answer:
{answer}

Now, you can create and collaborate with multiple experts to improve your answer. Therefore, please describe in as much detail as possible the different skills and focuses you need from multiple experts individually. 
We will provide each expert with the same information and query. However, please note that each profession has its own specialization, so you can assign each expert to just one sub-task to ensure a more refined response. 
We will relay their responses to you in turn, allowing you to reorganize them into a better answer.
Please note that the description should be narrated in the second person, for example: You are a XXX.

These are the descriptions of the experts you have created before for this task:
{description}

Therefore, please remember you should not repeatedly create the same experts as described above.
Now, you can give the description for a new expert (Please note that only be one, do not give multiple at one time):
"""
spy_meta_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "description"],
    template=SPY_INSTRUCTION_META,
)

SPY_INSTRUCTION_MULTI = """
{description}. 
{question}
You need to give reasons first and then give the answer with the format: \"Final Answer: <a single word from the word list>\" 
"""

spy_multi_agent_prompt = PromptTemplate(
    input_variables=["question", "description"],
    template=SPY_INSTRUCTION_MULTI,
)

SPY_INSTRUCTION_REFINE = """
{question}
This is your answer:
{old_answer}

You invite an expert whose description is: \"{description}\"
This expert also give his answer based on his own professional knowledge: {new_answer}.

Now you can refine your answer with his answer to better answer the question. 
Keep in mind that his answer may not be correct, so critically decide whether to accept his response or stick with your original one.
You need to give reasons first and then give the answer with the format: \"Final Answer: <a single word from the word list>\" 
Revised Answer:
"""

spy_refine_agent_prompt = PromptTemplate(
    input_variables=["question", "old_answer", "description", "new_answer"],
    template=SPY_INSTRUCTION_REFINE,
)

SPY_INSTRUCTION_SELF_REFINE = """

{question}
This is your answer:
{answer}

There is the suggestion from an assistant:
Suggestion: {feedback}

Now you can refine your answer with his suggestion to better answer the question. 
Keep in mind that his suggestion may not be correct, so critically decide whether to accept his response or stick with your original one.
You need to give reasons first and then give the answer with the format: \"Final Answer: <a single word from the word list>\" 
Revised Answer:
"""

spy_self_refine_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "feedback"],
    template=SPY_INSTRUCTION_SELF_REFINE,
)

SPY_INSTRUCTION_FEEDBACK = """
{question}
This is the answer from a student: {answer}.

Please do not refine the answer but give some insightful suggestions for the student to help him better answer the question.
Suggestion:
"""

spy_feedback_agent_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=SPY_INSTRUCTION_FEEDBACK,
)

from langchain.prompts import PromptTemplate

GUESS_INSTRUCTION_META = """
{question}
This is your answer:
{answer}

Now, you can create and collaborate with multiple experts to improve your answer. Therefore, please describe in as much detail as possible the different skills and focuses you need from multiple experts individually. 
We will provide each expert with the same information and query. However, please note that each profession has its own specialization, so you can assign each expert to just one sub-task to ensure a more refined response. 
We will relay their responses to you in turn, allowing you to reorganize them into a better answer.
Please note that the description should be narrated in the second person, for example: You are a XXX.

These are the descriptions of the experts you have created before for this task:
{description}

Therefore, please remember you should not repeatedly create the same experts as described above.
Now, you can give the description for a new expert (Please note that only be one, do not give multiple at one time):
"""
guess_meta_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "description"],
    template=GUESS_INSTRUCTION_META,
)

GUESS_INSTRUCTION_MULTI = """
{description}. 
{question}
You need to give reasons first and then give the answer with the format: \"Final Answer: <a comma-separated list of {n} words from the word list>\"  
"""

guess_multi_agent_prompt = PromptTemplate(
    input_variables=["question", "description", "n"],
    template=GUESS_INSTRUCTION_MULTI,
)



GUESS_INSTRUCTION_REFINE = """
{question}
This is your answer:
{old_answer}

You invite an expert whose description is: \"{description}\"
This expert also give his answer based on his own professional knowledge: {new_answer}.

Now you can refine your answer with his answer to better answer the question. 
Keep in mind that his answer may not be correct, so critically decide whether to accept his response or stick with your original one.
You need to give reasons first and then give the answer with the format: \"Final Answer: <a comma-separated list of {n} words from the word list>\"  
Revised Answer:
"""

guess_refine_agent_prompt = PromptTemplate(
    input_variables=["question", "old_answer", "description", "new_answer", "n"],
    template=GUESS_INSTRUCTION_REFINE,
)

GUESS_INSTRUCTION_SELF_REFINE = """

{question}
This is your answer:
{answer}

There is the suggestion from an assistant:
Suggestion: {feedback}

Now you can refine your answer with his suggestion to better answer the question. 
Keep in mind that his suggestion may not be correct, so critically decide whether to accept his response or stick with your original one.
You need to give reasons first and then give the answer with the format: \"Final Answer: <a comma-separated list of {n} words from the word list>\"  
Revised Answer:
"""

guess_self_refine_agent_prompt = PromptTemplate(
    input_variables=["question", "answer", "feedback", "n"],
    template=GUESS_INSTRUCTION_SELF_REFINE,
)

GUESS_INSTRUCTION_FEEDBACK = """
{question}
This is the answer from a student: {answer}.

Please do not refine the answer but give some insightful suggestions for the student to help him better answer the question.
Suggestion:
"""

guess_feedback_agent_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=GUESS_INSTRUCTION_FEEDBACK,
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