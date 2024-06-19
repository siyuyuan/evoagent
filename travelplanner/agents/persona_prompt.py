from langchain.prompts import PromptTemplate

PERSONA_GEN= """
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
Task: {query}
Participants except you: 
"""

persona_gen_agent_prompt=PromptTemplate(
                        input_variables=["query"],
                        template = PERSONA_GEN,
                        )




SUGGEST_GEN= """
You are a {persona}.

Given information: {text}
Query: {query}

Do not answer the query but give some suggestions for an AI assistant for better answer this query.
Suggestion:
"""

suggest_gen_agent_prompt=PromptTemplate(
                        input_variables=["text", "query", "persona"],
                        template = SUGGEST_GEN,
                        )


SSP_PLANNER_INSTRUCTION= """
You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

Given information: {text}
Query: {query}

These are suggestion from others:
{suggestion}

You can answer with these suggestions to better meet the query.
Travel Plan:
"""

ssp_planner_agent_prompt=PromptTemplate(
                        input_variables=["text", "query", "suggestion"],
                        template = SSP_PLANNER_INSTRUCTION,
                        )






SSP_PLANNER_INSTRUCTION_FEEDBACK = """
You are a {persona}
Given information: {text}
Query: {query}
This is the travel plan from a travel plan designer: {answer}.

Please do not refine the plan but give some insightful suggestions for the travel plan designer to help him better meet the user's query. If you think the answer is good and no suggestion, please only output \"Well Done!\"
Suggestion:
"""

ssp_feedback_planner_agent_prompt= PromptTemplate(
                        input_variables=["text", "query", "answer", "persona"],
                        template = SSP_PLANNER_INSTRUCTION_FEEDBACK,
                        )

SSP_PLANNER_INSTRUCTION_SELF_REFINE = """
You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

Given information: {text}
Query: {query}

This is your answer:
Travel Plan: {answer}.

There are suggestions from others:
{suggestion}

Now you can refine your answer with these suggestions to better meet the query. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example.
Refined Travel Plan: 
"""

ssp_self_refine_planner_agent_prompt=PromptTemplate(
                        input_variables=["text", "query", "answer", "suggestion"],
                        template = SSP_PLANNER_INSTRUCTION_SELF_REFINE,
                        )
