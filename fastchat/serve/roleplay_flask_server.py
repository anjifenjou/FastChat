import os
from fastchat import client
import json
from pprint import pformat
from flask import Flask, request, jsonify
from argparse import ArgumentParser
from datasets import load_dataset
import random
from json import JSONDecodeError
from nltk.tokenize import RegexpTokenizer

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
import re
from langdetect import detect, detect_langs, DetectorFactory

conv_map = {}  # a dictionary of active conversations


@app.route('/user_message', methods=['GET', 'POST'])
def user_message():
    """
    When in this function, the conversation has already been initialized
    :return:
    """
    json_query = request.get_json(force=True, silent=False)
    if isinstance(json_query, str):
        json_query = json.loads(json_query)
    user_utterance = json_query['user_utterance']
    sender_id = json_query['sender_id']
    ####################################################################################################################
    #                                    CONVERSATION INITIALIZATION MODULE
    ####################################################################################################################
    if sender_id not in conv_map.keys():
        bot_response = init_conversation(user_utterance=user_utterance, sender_id=sender_id)
        if not get_bool(args.shallow_roleplay):
            bot_response = clean_bot_response(bot_response)

    else:
        conv_map[sender_id]["messages"].append({'role': 'user', 'content': user_utterance})
        conv_map[sender_id]["num_user_turns"] += 1
        conversation = conv_map[sender_id]
        history = conversation["messages"]

        if not get_bool(args.shallow_roleplay):
            #  already taken into account in prompt builder
            # request_messages = [{"role": "request_type", "content": "roleplay_chat"},
            #                    {"role": "system", "content": prompt}]

            # TODO we should test search module here or later after model first response try (depending of what it says)
            ############################################################################################################
            #                                                  SEARCH MODULE
            ############################################################################################################
            search_decision = get_search_decision(user_utterance)
            knowledge_response = generate_knowledge_response(user_utterance) if search_decision else ""
            # request_messages += [{"role": "knowledge_response", "content": knowledge_response}]

            ############################################################################################################
            #                                         ADDING ASSISTANT PERSONA TRAITS IN THE REQUEST
            ############################################################################################################
            assistant_persona = conv_map[sender_id]["assistant_persona"]
            assistant_name = conv_map[sender_id].get("assistant_name", "")
            # request_messages += [
            #     {"role": "assistant_persona", "content": '||'.join(conv_map[sender_id]["assistant_persona"])},
            #     {"role": "assistant_name", "content": conv_map[sender_id].get("assistant_name", "")}]

            # "build" persona based on the prompt type ? should be done in prompt building module

            ############################################################################################################
            #                          ADDING USER PERSONA TRAITS (Built from conversation by the agent)
            ############################################################################################################
            user_persona = conversation.get("user_persona", [])
            # if user_persona:
            #     request_messages.append({"role": "user_persona", "content": '||'.join(user_persona)})

            ############################################################################################################
            #                            ACCESSING MEMORY IF CONDITION REACHED (modify history then)
            ############################################################################################################
            # if approx_new_prompt_length + args.max_new_tokens + 8 > args.prompt_length_threshold:
            # access memory when reaching condition

            approximate_input_tokens = approx_tokens_per_word * words_count(user_utterance)
            # TODO import the LLaMA tokenizer here (to have exact count)
            approx_new_prompt_length = conv_map[sender_id]["last_output_size"] + approximate_input_tokens

            # + new_user_persona_length
            if approx_new_prompt_length + args.max_new_tokens + 8 > 2048:  # access memory when reaching condition
                # 2048 is the max context length of LLaMA # TODO may be consider a while loop ?
                conv_map[sender_id]['num_memory_access'] += 1

                if not conv_map[sender_id]["memory"]:  # No memory built yet => build it up to the current point
                    start = 0  # if there is no memory then we are at the start of the conversation
                    stop = len(history[:-1])  # to keep at least the last user input
                    conv_map[sender_id]["memory"].append(generate_memory(history[start: stop],
                                                                         start=start, stop=stop))
                request_memory_content, history = \
                    get_memory_content(sender_id, memory_index=conv_map[sender_id]['num_memory_access'])

                # request_messages.append({"role": "memory", "content": '||'.join(request_memory_content)})

            elif conv_map[sender_id]['num_memory_access'] > 0:
                # send the memories already accessed in preceding turns if the condition for new history is not reached
                request_memory_content, history = \
                    get_memory_content(sender_id, memory_index=conv_map[sender_id]['num_memory_access'])

                # request_messages.append({"role": "memory", "content": '||'.join(request_memory_content)})

        ################################################################################################################
        #                                    ADD HISTORY AND SEND REQUEST TO API
        ################################################################################################################
        # request_messages += history

        prompt, history = build_prompt(prompt_type='full', assistant_persona=assistant_persona,
                                       assistant_name=assistant_name, user_persona=user_persona,
                                       memory=request_memory_content, messages=history,
                                       knowledge_response=knowledge_response, style="")

        request_messages = [{"role": "system", "content": prompt}] + history
        # Send a request to the API
        response_with_complete_sentence = None
        while response_with_complete_sentence is None:
            completion = client.ChatCompletion.create(
                model="vicuna-13b-v1.1",
                messages=request_messages,
                max_tokens=args.max_new_tokens,  # added as the model tends to talk too much
            )

            uncleaned_response = completion.choices[0].message.content
            bot_response = clean_bot_response(uncleaned_response)
            response_with_complete_sentence = remove_incomplete_sentence(bot_response)

        bot_response = response_with_complete_sentence

        tokens_usage = completion.choices[0].usage  # usage is now in choices
        print(tokens_usage)

        print(f"Uncleaned bot response: {uncleaned_response}")
        # bot_response = clean_bot_response(bot_response)
        ################################################################################################################
        #                                            UPDATE CONVERSATION HISTORY
        ################################################################################################################
        conv_map[sender_id]["messages"].append({'role': 'assistant', 'content': bot_response})
        conv_map[sender_id]["last_output_size"] = tokens_usage["total_tokens"]
        history = conv_map[sender_id]["messages"]

        ################################################################################################################
        #                                           USER PERSONA BUILDING MODULE
        thrshld_len = len(history) + 1 if conv_map[sender_id]["chosen_persona"] else len(history)
        # in case of chosen persona (user assigning to assistant, their first message is not added to history)
        ################################################################################################################
        if history and thrshld_len % (num_turn_threshold * 2) == 0:
            user_utterances = [message for message in history[- num_turn_threshold * 2:] if message["role"] == 'user']
            conv_map[sender_id]["user_persona"].append(generate_user_persona(user_utterances))

        ################################################################################################################
        #                                               MEMORY BUILDING MODULE
        ################################################################################################################
        if history and thrshld_len % num_turn_threshold == 0:  # may be improve this based on Few-Shot Bot
            if conv_map[sender_id]["memory"]:
                start = conv_map[sender_id]["memory"][-1]["stop"]
            else:
                start = 0
            stop = start + num_turn_threshold
            conv_map[sender_id]["memory"].append(generate_memory(history[start: stop],
                                                                 start=start, stop=stop))

    # print(bot_response)
    # print()
    return jsonify(
        {
            'persona': ' || '.join(assistant_persona),
            'messages': conv_map[sender_id]["messages"],
            'chatbot_utterance': bot_response,
            'num_user_turns': conv_map[sender_id]["num_user_turns"],
        }
    )


@app.route('/kill_conversation', methods=['GET', 'POST'])
def kill_conversation():
    json_query = request.get_json(force=True, silent=False)
    if isinstance(json_query, str):
        json_query = json.loads(json_query)
    sender_id = json_query['sender_id']

    if sender_id in conv_map.keys():
        del conv_map[sender_id]

    return jsonify(
        {
            'uuid': None,
            'messages': [],
            'chatbot_utterance': "",
        }
    )


# system: str, seps: list = [" ", "</s>"],
def build_prompt(prompt_type: str = 'full', assistant_name: str = None,
                 assistant_persona: list[str] = None, user_persona: list[str] = None, memory: list[str] = None,
                 messages: list[str] = [], knowledge_response: str = None, style: str = None, ):
    """
    Args:
        prompt_type: a string describing the type of prompt selected by the RL.
                    Can be 'full', 'inc' or 'wind'. default 'full'
        system: A string describing how the system should globally behave
        seps: default separator of the backbone LLM (here Vicuna by default)
        assistant_name: a string decribing the name of the assistant throughout the conversation
        assistant_persona: a list of string each describing a particular traits the assistant should roleplay
        user_persona: a list of string describing the user and derived from the previous utterances
        memory: a short summary of older dialogue episodes (here used when the prompt is too long and the messages are
                truncated
        messages: history of the conversation. If memory is used it doesn't include older messages that are summarized
        knowledge_response: a string of external information to ground the answer (this is not the retrieved documents
                            or memory but a sentence that gathers the relevant information to ground the answer)
        style: the tone/style of writing that the response should convey ( sad, happy, joy etc... it is actually Image Chat that has style)
        #TODO: it could be interesting to dynamically (automatically determine the writing style ?)
        #TODO: it should be better evaluated: rather than just engagingness, assess persona consistency, style etc
         #TODO persona and style Adjustment to enhance open domain conversation
    Return:
        ret: a string corresponding to the built prompt. In the case external information/instruction should be added
            in between speakers back-and-fourths we return the last users messages modified to add them
    """

    separation = "." + f"{sep}"

    if args.shallow_roleplay:

        ret = f"""
{shallow_desc + sep} 

{"The following sentences describe assistant personality and background: " +
 " ".join(assistant_persona) + sep2 if assistant_persona else ""}  

{"Complete the following conversation as the assistant with the described character would with a short response in French: "
        if len(messages) > 1 else
        "Start a conversation as the assistant with the described character would with a short response in French: "}
"""

    else:  # advanced and verbose prompt

        if prompt_type == 'inc':  # need to keep track of already exchanged persona traits/statements aka profiles
            assistant_persona = [assistant_persona[0]]  # to display only one

        ret = f"""
{system_desc + sep}
{"You SHALL ALWAYS respond in French." + sep}
{"You ACT as a regular person. You are NOT a chatbot. You are NOT an assistant."}
{"You have a name. "}{"Your name is " + assistant_name + "." + sep if assistant_name else
        "If asked you SHALL ALWAYS choose and give a REAL name adapted to your personality described below."}
"""

        if assistant_persona:
            ret += f"""
{"YOUR personality is: " + separation.join(assistant_persona) if assistant_persona else ""}
{"Remember, you always stay on character. You are the character described above." if assistant_persona else ""}   

"""

        if user_persona:
            ret += f"""
{"You know this about the user you are talking to: " + separation.join(user_persona)
 + ". Use it to adapt your conversation to the user" if user_persona else ""}
 
"""

        if memory:
            ret += f"""
{"Here is a summary of previous sessions of this conversation to help you remember what has been said: " +
 sep.join(memory) if memory else ""}    
"""  # in SOTA this is not how memory is used, stored in a based and queried when necessary using retrieval & matching
        # technique

        ret += f"""
{"Complete the following conversation with a short and precise sentence as your character would.  "
 "Always speak with new and unique messages that haven't been said in the conversation :"
        if len(messages) > 1
        else "Start a conversation in French, empathically as your character would. Write your input only, not the user's "
             "response. Do not offer your help, be nice you are talking to a user who just wants to have a discussion. "
             "You can limit yourself to a greeting:"}

"""
        if knowledge_response:
            # should we add new line here ? # may be add new line after each speaker response?
            # messages[-1] = messages[-1] + "\n" + f" Voici des informations supplémentaires récupérées sur Internet" \
            #                f" à propos du dernier message de l'utilisateur:" \
            #                f"{knowledge_response}. Utilise cela pour construire ta réponse. "
            # it may be better to be less verbose on this knowledge introduction, especially that it some information inside
            # conversation

            messages[-1] = messages[-1][
                               'content'] + "\n" + f"Connaissances supplémentaires pour la reponse :{knowledge_response}.\n"
            # last message now includes additional knowledge

        if style:  # should be the last directive before model response # if in another langugae than response language
            # it can be misleading
            messages[-1] = messages[-1]['content'] + "\n" + f"Réponds avec le style suivant:{style}. \n"
            # in this way also at next turn user messages would be as if there were new informations

    return ret, messages


def words_count(utterance, nltk_tokenizer=RegexpTokenizer(r'\w+')):
    if isinstance(utterance, str):
        return len(nltk_tokenizer.tokenize(utterance))
    else:
        print("The utterance is not a string")
        return 0


def detect_majority_language(text):
    """
    Detect if there is more than one language, and french is not most present regenerate
    """
    DetectorFactory.seed = 0
    lang, prob = '', 0
    text = text.strip()
    if text:  # to avoid errors when the string is empty
        languages = detect_langs(text)
        majority_language = max(languages, key=lambda lang: lang.prob)
        lang, prob = majority_language.lang, majority_language.prob
        print(f'Language is: {lang} and probability is: {prob:.2f}')
    return lang, prob


def init_conversation(sender_id, user_utterance):
    """
    This function initialize the conversation metadata (persona, name, etc.)
    Either from PersonaChat Dataset or from the user (retrieved via a specific designed prompt on the user's first input
    sent to the instruction model).
    :return:
        A json containing new bot message based on persona.
    """
    assistant_persona = random.choice(pchat_personalities)
    assistant_name = ""
    # chosen_persona = get_desired_persona(user_utterance=user_utterance)  # case when user want to assign a persona.
    chosen_persona = False
    text = f"Persona is randomly assigned from personaChat: {'|'.join(assistant_persona)} "
    if chosen_persona:
        assistant_persona = chosen_persona.get("persona", assistant_persona)
        text = f"The user assigned the following persona: {'|'.join(assistant_persona)} "
        assistant_name = chosen_persona.get("name", assistant_name)
    print(text)

    conversation_dict = {"assistant_persona": assistant_persona,
                         "user_persona": [],
                         "memory": [],
                         "assistant_name": assistant_name,
                         "messages": [],
                         "last_output_size": 0,
                         "num_memory_access": 0,
                         # "chosen_persona": chosen_persona is not None
                         "chosen_persona": chosen_persona,
                         "num_user_turns": 0}

    conv_map[sender_id] = conversation_dict
    # TODO: put the prompt type in flask args or in the request sent to the flask
    prompt, _ = build_prompt(prompt_type='full', assistant_persona=assistant_persona, assistant_name=assistant_name,
                             user_persona=conversation_dict.get("user_persona", []), style="")

    request_msg = [{"role": 'system', 'content': prompt}]
    if not chosen_persona:
        request_msg += [{"role": "user", "content": user_utterance}]
        conv_map[sender_id]["num_user_turns"] += 1
    # if the first user_utterance was to define the persona, we don't send it in the request and the
    # assistant will start the conversation
    response_with_complete_sentences = None

    while response_with_complete_sentence is None:
        lang, prob = '', 0
        while lang != 'fr' or prob < 0.60:
            completion = client.ChatCompletion.create(
                model="vicuna-13b-v1.1",
                messages=request_msg,
                # n=3 if chosen_persona else 1,
                max_tokens=50  # force the bot to start with a short message
            )
            uncleaned_first_message = completion.choices[0].message.content
            print(uncleaned_first_message)
            bot_first_message = clean_bot_response(uncleaned_first_message)
            lang, prob = detect_majority_language(bot_first_message)
        response_with_complete_sentence = remove_incomplete_sentence(bot_first_message)

    bot_first_message = response_with_complete_sentence

    tokens_usage = completion.choices[0].usage
    conv_map[sender_id]["messages"] = [] if chosen_persona else [{'role': 'user', 'content': user_utterance}]
    conv_map[sender_id]["messages"] += [{'role': 'assistant', 'content': bot_first_message}]
    conv_map[sender_id]["last_output_size"] = tokens_usage["total_tokens"]

    print(f"Uncleaned bot first message: {uncleaned_first_message}")

    return bot_first_message
    # jsonify({
    #    'persona': ' || '.join(conv_map[sender_id]["assistant_persona"]),
    #    'messages': conv_map[sender_id]["messages"],
    #    'chatbot_utterance': bot_first_message,
    # })


def remove_incomplete_sentence(text):
    text = text.strip()

    if text[-1] not in [".", "!", "?"]:
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s*'
        phrases = re.split(pattern, text)
        print(phrases)
        if len(phrases) > 1:
            phrases_sans_derniere = ' '.join(phrases[:-1])
            text = phrases_sans_derniere
        else:
            return None
    return text


def clean_bot_response(bot_response):
    # TODO: report in  the paper, Shall we detect and regenerate or just remove as currently
    bot_response = bot_response.strip()
    string_to_remove = [r'^\s*\([^)]*\)\s*',  # (text_au_debut) souvent des didascalies
                        r'\s*\([^)]*\)\s*[.,]?\s*$',  # (text_a_la_fin) souvent des traductions,
                        r'^\s*\[[^\]]*\]\s*',  # [text_debut]
                        r'\s*\[[^\]]*\]\s*[.,]?\s*$',  # [text_fin]
                        "En tant que personnage fictif\\s+?,", "un assistant intelligent",
                        "en tant qu['e] \\w+\\s?,",
                        "en tant qu['e]\\s*?\\S[\\S\\s]*?,"]

    expression_regulieres = re.compile("|".join(string_to_remove), re.IGNORECASE)
    quotes = [r'^"(.*)"$', r"^'(.*)'$", r"^'''(.*)'''$"]
    quotes = re.compile("|".join(quotes), re.IGNORECASE)
    cleaned_bot_response = expression_regulieres.sub("", bot_response)
    cleaned_bot_response = quotes.sub(r'\1', cleaned_bot_response)
    return cleaned_bot_response


def get_bool(toto):
    return str(toto).lower().strip() == "true"


def get_desired_persona(
        user_utterance):  # Dire à l'utlisateur de donner des mots clés pour communiquer le personnage
    # TODO: may be return string directly instead of list for persona
    get_persona_prompt = f"""
    Dans le texte ci-dessous, vérifie si des éléments par rapport à un nom ou des traits de personnalité sont donnés. 
    Si c'est le cas retourne les informations au format json avec les champs suivants: name (si applicable) et persona (une liste de chaines de caractères décrivant la personalité).
    Si ce n'est pas le cas retourne "None".

    Texte:  
        "{user_utterance}"
    """  # positivity in the prompt

    # if not fully working can bypass it by always returning None
    completion = client.ChatCompletion.create(
        model="vicuna-13b-v1.1",
        messages=[{"role": "user", "content": get_persona_prompt}]
        # {"role": "request_type", "content": "submodule_chat"}],
    )
    persona_response = completion.choices[0].message.content
    print(f"The persona response is: {persona_response}")
    if persona_response.strip().lower() == 'none':
        return None
    else:  # supposedly we have the json here
        try:
            chosen_persona = json.loads(persona_response)
            if not isinstance(chosen_persona, dict):  # The case when it is a long string for stating None
                return None
            else:
                if not isinstance(chosen_persona["persona"], list):
                    if isinstance(chosen_persona["persona"], str):
                        if chosen_persona["persona"].strip().lower() == "none":
                            return None
                        else:
                            chosen_persona["persona"] = [chosen_persona["persona"]]
                    else:
                        return None
        except JSONDecodeError:
            return None
        return chosen_persona


def generate_user_persona(user_utterances):
    formatted_user_utterances = ""
    for i, message in enumerate(user_utterances):
        formatted_user_utterances += message["role"] + ": " + message["content"] + "\n"

    user_persona_gen_prompt = f"""
Rédige un résumé très court de la personnalité de l'utilisateur suivant en une phrase courte.
Le résumé ne doit pas comporter le mot "utilisateur", uniquement sa description.  
Le résumé est limité à 10 mots. 

{formatted_user_utterances}
"""
    completion = client.ChatCompletion.create(
        model="vicuna-13b-v1.1",
        messages=[{"role": "user", "content": user_persona_gen_prompt}, ]
        # {"role": "request_type", "content": "submodule_chat"}],
        # n=3  # TODO: think of generating several choices for submodules request and select among them?
    )
    user_persona = completion.choices[0].message.content
    return user_persona


def generate_memory(episode, start, stop,
                    seps=[" ", "</s>"]):  # memory = {"start": 0, "stop":4, content:"le resume des tours 0 à 3"
    formatted_episode = ""
    # here we build conversation history
    for i, message in enumerate(episode):
        if message['role'] == 'assistant':
            formatted_episode += message["role"] + ": " \
                                 + message["content"] \
                                 + seps[0] + "\n"
        elif message['role'] == 'user':
            formatted_episode += message["role"] + ": " \
                                 + message["content"] + seps[0] + "\n"

    memory_generation_prompt = f"""
Rédige un COURT résumé  de la conversation suivante entre un assistant et un utilisateur en une phrase. 
Le résumé doit comporter des informations pertinentes et ne doit pas commencer par "dans cette conversation"
Le résumé doit faire maximum 100 mots.

{formatted_episode}
"""
    completion = client.ChatCompletion.create(
        model="vicuna-13b-v1.1",
        messages=[{"role": "user", "content": memory_generation_prompt}, ]
        # {"role": "request_type", "content": "submodule_chat"}],
    )
    new_memory = completion.choices[0].message.content
    return {"start": start, "stop": stop, "content": new_memory}


def get_memory_content(sender_id, memory_index):
    request_memory_content = [memory["content"] for memory in conv_map[sender_id]["memory"][:memory_index]]
    new_history_start = conv_map[sender_id]["memory"][:memory_index][-1]["stop"]
    # new_history_start -= 2   # to make sure we have at least
    history = conv_map[sender_id]["messages"][new_history_start:]
    if len(history) < 2:
        history = conv_map[sender_id]["messages"][new_history_start - 2:]
    return request_memory_content, history


def get_search_decision(user_utterance):
    # TODO add a designed search decision prompt later
    search_decision_prompt = f"""
Check the following text to see if any information is requested.
Check if an entity you don't know is mentioned.
Check if an information from the period beyond your knowledge limit (2021) is asked.

If it is the case then generate a query in french for a search based on the text. In this case return only the content of the query preceded by Search. Required format: "Search: query".

If it's not the case then return only: no_search

text=
"{user_utterance}" 
    """

    # completion = client.ChatCompletion.create(
    #     model="vicuna-13b-v1.1",
    #     messages=[{"role": "user", "content": search_decision_prompt},
    #              {"role": "request_type", "content": "submodule_chat"}],
    # possibility to generated several choice and perform majority vote on search vs no search?
    # )
    # search_decison = completion.choices[0].message.content

    # if search_decison.lower().strip() == 'no_search':
    #    return False
    # else:
    #    query = search_decison.split(":")[-1]
    #    # TODO SEND THE QUERY TO A SEARCH SERVER
    #    return query
    return False


def generate_knowledge_response(user_utterance):  # Call a search server
    # TODO add the query generation prompt and a call to as search engine later
    knowledge_response = ""
    return knowledge_response


def init_app_parameters():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/Volumes/Crucial/Thesis/Datasets/PersonaChat/personachat_self_original.json",
                        help="Path to  PersonaChat dataset to get personalities")
    parser.add_argument("--prompt_length_threshold", type=int, default=1024,  # may be reduce it
                        help="Number of token we don't want to exceed. "
                             "If Reached, generate and access memory and reduce the history in the prompt")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Maximum number of generated tokens")  # TODO adapt it dynamically with user input
    parser.add_argument("--num_turn_threshold", type=int, default=4,
                        help="Number of turn after which we generate new memory and store it. Memory is not necessarily"
                             " used straight after but can be accessed  later based on token threshold ")
    parser.add_argument("--approx_tokens_per_word", type=float, default=1.5,  # may be reduce it
                        help="Estimate of the average number of tokens per words to comput approx tokens length"
                             "of the user input")
    parser.add_argument("--search_server", type=str,
                        help="Address of the search server, use to query the web when needed")
    parser.add_argument("--api_address", type=str,
                        help="Address of API to which the request are sent")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Address of the flask_server host")
    parser.add_argument("--port", type=str, default=None,
                        help="port of the flask_server ")
    parser.add_argument("--shallow_roleplay", type=str, default="False",
                        help="Whether or not to run a shallow prompt version of the roleplay.")
    args = parser.parse_args()
    # PERSONALITIES
    data_path = args.data_path
    raw_dataset = load_dataset("json", data_files=data_path, field="train")
    personalities = raw_dataset["train"]["personality"]

    os.environ['FASTCHAT_BASEURL'] = args.api_address
    client.set_baseurl(os.getenv("FASTCHAT_BASEURL"))

    print("Arguments: %s", pformat(args))
    return args, personalities


if __name__ == "__main__":  # setting up args
    args, pchat_personalities = init_app_parameters()
    # system_desc = "A chat between a curious user and an artificial intelligence assistant. " \
    #             "The assistant is engaging, empathetic, and role plays as the character described below. " \
    #              "The assistant gives helpful, detailed and polite an answers to user's questions " \
    #              "and the assistant feels free to ask questions to the user and  make jokes."`
    # system_desc = "Un dialogue entre un utilisateur curieux et un assistant intelligent." \
    #                  "L'assistant est impliqué, empathique et joue le rôle du personnage décrit ci-dessous. " \
    #                  "L'assistant donne des réponses utiles, détaillées et polies aux questions de l'utilisateur." \
    #                  "L'assistant est libre de poser des questions à l'utilisateur et de faire des blagues."

    # system_desc = "Joue le role du personnage décrit dans les lignes suivantes. Tu conserves toujours cette personnalité." \
    #              "Tu es impliqué, empathique, tu donnes des réponses utiles, courtes et simples à l'utilisateur" \
    #              "Tu poses des questions à l'utlisateur par rapport à ce qu'il dit ou pour en savoir plus sur lui. " \
    #              "Tu fais des blagues."

    system_desc = "Role play as the character described in the following lines. You always stay in character." \
                  "You are engaging, empathetic, you give useful, short and simple answers to the user." \
                  "You ask the user questions about what they are saying or to find out more about them." \
                  "You make jokes."

    shallow_desc = "A chat between a curious user and an artificial intelligence assistant. " \
                   "The assistant gives helpful, detailed and polite an answers to user's questions." \
                   "The assistant role plays as the character described below. "
    # "The assistant always speak in French" \
    sep, sep2 = [" ", "</s>"]
    prompt_length_threshold = args.prompt_length_threshold
    num_turn_threshold = args.num_turn_threshold
    approx_tokens_per_word = args.approx_tokens_per_word

    app.run(host=args.host if args.host else "0.0.0.0",
            port=args.port if args.port else 5008)
