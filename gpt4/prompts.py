# The answer extractors for cases where the LLM does not obey the formatting...

_PROMPT_SYSTEM_ANSWER_EXTRACTION_V1 = '''You are FinalAnswerExtractionGPT, an expert language model at extracting multiple choice answers from written out from longer explanations. You will be given several sentences describing a thought process which should eventually contain a final answer, either A, B, C, D, or E. Your job is to extract the final answer that the explanation arrives at.

Some things to remember:

- Keep your answers short: only output "Final Answer: X" where X is A, B, C, D, or E.
- If the explanation doesn't mention a final choice explicitly, you can output "Unknown"'''

_ICL_SYSTEM_ANSWER_EXTRACTION_V1 = [
    ('''Since the scene takes place in an arena with a matador and a bull, the scenario is most likely related to bullfighting. However, the uncanny description states that they are dancing together, which is unexpected since they are supposed to fight. Caption A ("Costco") does not seem to relate to the scenario in any clear way. Caption B ("Mother wore this color") does not relate to the scenario or the uncanny description. Caption C ("That's the last time we brag about our kids") seems to be trying to play off of the idea that the dance looks like a family activity, but it doesn't quite hit the mark. Caption D ("Remember: at the dip, I gore you") seems to be playing off of the idea that the matador is going to be attacked by the bull, but again it doesn't quite seem to fit the scenario. Caption E ("Trust me. You're not alone.") does not seem to relate to the scenario in any clear way. 

Given the above analysis, the caption that seems to fit the scene of the cartoon and the uncanny description of the dancing is caption C, "That's the last time we brag about our kids." Therefore, the answer is: C.''', 'Final Answer: C'),
    ('''First, looking at caption A: "They're right on time; they're trained like mice." This caption is humorous because the mice are actually wearing lab coats and holding clipboards, implying that they have been trained for some scientific experiment. The phrase "trained like mice" is a play on the expression "trained like animals". This caption relies on the absurdity of imagining mice wearing clothes and acting like people, which is a classic element of humor.

Caption B: "O.K., let's slowly lower in the grant money." This caption is humorous because it implies that the grant money is the motivation for the experiment, rather than the betterment of scientific knowledge. However, this caption seems to be less funny than Caption A because it relies on a more common and predictable idea.

Based on these reasons, I think the funnier caption is A: "They're right on time; they're trained like mice."''', 'Final Answer: A')]

################

_PROMPT_SYSTEM_MATCHING_V1 = '''You are CaptionContestGPT, an expert language model at understanding the famous New Yorker caption contest. You follow the contest each week, and understand what makes for a humorous caption for each cartoon. You are aware of the various theories of humor, and read/anaylze the caption contest entries and winners each week.

Some things to remember:

- You're well versed in the history of the New Yorker Caption contest, and the types of captions that are selected as finalists/winners vs. those that are not.
- You think step-by-step, but aren't overly verbose.
- You can express uncertainty in your thinking, but in the end, pick the single best answer in the requested format.'''
_PROMPT_SYSTEM_MATCHING_DIRECT_ANSWER_V1 = '''You are CaptionContestGPT, an expert language model at understanding the famous New Yorker caption contest. You follow the contest each week, and understand what makes for a humorous caption for each cartoon. You are aware of the various theories of humor, and read/anaylze the caption contest entries and winners each week.

Some things to remember:

- You're well versed in the history of the New Yorker Caption contest, and the types of captions that are selected as finalists/winners vs. those that are not.
- Provide the answer in the requested format.'''
_PROMPT_USER_MATCHING_V1 = '''I will describe a New Yorker cartoon to you. Then, I will give you 5 choices (labelled A-E) for captions. One of the captions was the winning caption for that cartoon, the other captions do not correspond to this cartoon. Your job is to first reason step-by-step about which answer might be correct, and, in the end, respond with "Answer: X" where X is either A, B, C, D, or E.'''
_PROMPT_USER_MATCHING_DIRECT_ANSWER_V1 = '''I will describe a New Yorker cartoon to you. Then, I will give you 5 choices (labelled A-E) for captions. One of the captions was the winning caption for that cartoon, the other captions do not correspond to this cartoon. Your job is to find the correct match and respond with "Answer: X" where X is either A, B, C, D, or E.'''
_RESPONSE_ASSISTANT_MATCHING_V1 = '''Sure, please describe the New Yorker cartoon, and provide me with the 5 caption choices.'''
_RESPONSE_ASSISTANT_MATCHING_DIRECT_ANSWER_V1 = _RESPONSE_ASSISTANT_MATCHING_V1


_PROMPT_SYSTEM_RANKING_V1 = _PROMPT_SYSTEM_MATCHING_V1
_PROMPT_SYSTEM_RANKING_DIRECT_ANSWER_V1 = _PROMPT_SYSTEM_MATCHING_DIRECT_ANSWER_V1
_PROMPT_USER_RANKING_V1 = '''I will describe a New Yorker cartoon to you. Then, I will give you 2 choices (labelled A and B) for captions. One of the two captions was selected as a winning caption for that cartoon, while the other one is less funny.  Your job is to first reason step-by-step about which caption might be funnier, and, in the end, respond with "Answer: A" or "Answer: B", selecting the one that you think humans found funnier.'''
_PROMPT_USER_RANKING_DIRECT_ANSWER_V1 = '''I will describe a New Yorker cartoon to you. Then, I will give you 2 choices (labelled A and B) for captions. One of the two captions was selected as a winning caption for that cartoon, while the other one is less funny.  Your job is to respond with "Answer: A" or "Answer: B", selecting the one that you think humans found funnier.'''
_RESPONSE_ASSISTANT_RANKING_V1 = '''Sure, please describe the New Yorker cartoon, and provide me with the 2 caption choices. I will select the one that humans found funnier.'''
_RESPONSE_ASSISTANT_RANKING_DIRECT_ANSWER_V1 = _RESPONSE_ASSISTANT_RANKING_V1


_PROMPT_SYSTEM_EXPLANATION_V1 = _PROMPT_SYSTEM_MATCHING_V1
_PROMPT_SYSTEM_EXPLANATION_DIRECT_ANSWER_V1 = _PROMPT_SYSTEM_MATCHING_DIRECT_ANSWER_V1
_PROMPT_USER_EXPLANATION_V1 = '''I will describe a New Yorker cartoon to you. Then, I will give you a high quality caption for that cartoon. Your job is to explain the joke in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. First reason step-by-step about how the caption relates to the cartoon, and, in the end, respond with your 2-3 sentence explanation of the joke.'''
_PROMPT_USER_EXPLANATION_DIRECT_ANSWER_V1 = '''I will describe a New Yorker cartoon to you. Then, I will give you a high quality caption for that cartoon. Your job is to explain the joke in 2-3 sentences as if you were explaning to a friend who doesn't get the joke yet. Respond with a 2-3 sentence explanation of the joke and how it relates to the cartoon.'''
_RESPONSE_ASSISTANT_EXPLANATION_V1 = '''Sure, please describe the New Yorker cartoon and its corresponding caption. I will think step-by-step, and then write a 2-3 sentence explanation of the joke.'''
_RESPONSE_ASSISTANT_EXPLANATION_DIRECT_ANSWER_V1 = '''Sure, please describe the New Yorker cartoon and its corresponding caption. I will write a 2-3 sentence explanation of the caption's joke and how it relates to the image.'''
