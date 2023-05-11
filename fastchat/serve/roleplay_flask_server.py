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
        if get_bool(args.shallow_roleplay):
            ret = f"""
{shallow_desc + sep}

{"The following sentences describe assistant personality and background: " +
" ".join(conv_map[sender_id]["assistant_persona"]) + sep2 if conv_map[sender_id]["assistant_persona"]
else ""}

{"Complete the following conversation as the assistant with the described character would with a short response: "}

"""
            request_messages = [{"role": "system", "content": ret}]
        else:
            conv_map[sender_id]["messages"].append({'role': 'user', 'content': user_utterance})
            conversation = conv_map[sender_id]
            history = conversation["messages"]
            request_messages = [{"role": "request_type", "content": "roleplay_chat"},
                                {"role": "system", "content": system_desc}]

            # TODO we should test search module here or later after model first response try (depending of what its says)
            ################################################################################################################
            #                                                  SEARCH MODULE
            ################################################################################################################
            search_decision = get_search_decision(user_utterance)
            if search_decision:
                knowledge_response = generate_knowledge_response(user_utterance)
                request_messages += [{"role": "knowledge_response", "content": knowledge_response}]

            ################################################################################################################
            #                                         ADDING ASSISTANT PERSONA TRAITS IN THE REQUEST
            ################################################################################################################
            request_messages += [
                {"role": "assistant_persona", "content": '||'.join(conv_map[sender_id]["assistant_persona"])},
                {"role": "assistant_name", "content": conv_map[sender_id].get("assistant_name", "")}]

            ###############################################################################################################
            #                          ADDING USER PERSONA TRAITS (Built from conversation by the agent)
            ################################################################################################################
            user_persona = conversation.get("user_persona", [])
            if user_persona:
                request_messages.append({"role": "user_persona", "content": '||'.join(user_persona)})

            ###############################################################################################################
            #                               ACCESSING MEMORY IF CONDITION REACHED (modify history then)
            ###############################################################################################################
            if conv_map[sender_id]["memory"]:  # if there is no memory yet the prompt will just be trimmed by the worker
                approximate_input_tokens = approx_tokens_per_word * words_count(user_utterance)
                # TODO import the LLaMA tokenizer here (to have exact count)
                approx_new_prompt_length = conv_map[sender_id]["last_output_size"] + approximate_input_tokens
                # + new_user_persona_length

                if approx_new_prompt_length + args.max_new_tokens + 8 > 2048:  # access memory when reaching condition
                    # 2048 is the max context length of LLaMA # TODO may be consider a while loop ?
                    conv_map[sender_id]['num_memory_access'] += 1
                    request_memory_content, history = \
                        get_memory_content(sender_id, memory_index=conv_map[sender_id]['num_memory_access'])

                    request_messages.append({"role": "memory", "content": '||'.join(request_memory_content)})

                # send the memories already accessed in preceding turn if the condition for new history is not reached
                elif conv_map[sender_id]['num_memory_access'] > 0:
                    request_memory_content, history = \
                        get_memory_content(sender_id, memory_index=conv_map[sender_id]['num_memory_access'])

                    request_messages.append({"role": "memory", "content": '||'.join(request_memory_content)})

        ################################################################################################################
        #                                    ADD HISTORY AND SEND REQUEST TO API
        ################################################################################################################
        request_messages += history

        # Send a request to the API
        completion = client.ChatCompletion.create(
            model="vicuna-13b-v1.1",
            messages=request_messages,
            max_tokens=args.max_new_tokens,  # added as the model tends to talk too much
        )

        tokens_usage = completion.choices[0].usage  # usage is now in choices
        print(tokens_usage)
        bot_response = completion.choices[0].message.content
        print(f"Uncleaned bot response: {bot_response}")
        bot_response = clean_bot_response(bot_response)
        ################################################################################################################
        #                                            UPDATE CONVERSATION HISTORY
        ################################################################################################################
        conv_map[sender_id]["messages"].append({'role': 'assistant', 'content': bot_response})
        conv_map[sender_id]["last_output_size"] = tokens_usage["total_tokens"]
        history = conv_map[sender_id]["messages"]

        ################################################################################################################
        #                                           USER PERSONA BUILDING MODULE
        thrshld_len = len(history) + 1 if conv_map[sender_id]["chosen_persona"] else len(history)
        # in case of chosen persona, user first message is not added to history
        ################################################################################################################
        if history and thrshld_len % (num_turn_threshold * 2) == 0:
            # build user persona in this episode:
            # stop = len(history)  # // (num_turn_threshold * 2)
            # start = len(history) - num_turn_threshold * 2
            user_utterances = [message for message in history[- num_turn_threshold * 2:] if message["role"] == 'user']
            conv_map[sender_id]["user_persona"].append(generate_user_persona(user_utterances))

        ################################################################################################################
        #                                               MEMORY BUILDING MODULE
        ################################################################################################################
        if history and thrshld_len % num_turn_threshold == 0:
            if conv_map[sender_id]["memory"]:
                start = conv_map[sender_id]["memory"][-1]["stop"]
            else:
                start = 0
            stop = start + num_turn_threshold
            conv_map[sender_id]["memory"].append(generate_memory(history[start: stop],
                                                                 start=start, stop=stop))

    # print(bot_response)
    print()

    return jsonify(
        {
            'persona': ' || '.join(conv_map[sender_id]["assistant_persona"]),
            'messages': conv_map[sender_id]["messages"],
            'chatbot_utterance': bot_response,
        }
    )


def words_count(utterance, nltk_tokenizer=RegexpTokenizer(r'\w+')):
    if isinstance(utterance, str):
        return len(nltk_tokenizer.tokenize(utterance))
    else:
        print("The utterance is not a string")
        return 0


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
    # chosen_persona = get_desired_persona(user_utterance=user_utterance)  # The case of user desired persona.
    chosen_persona = False
    text = f"Persona is randomly assigned from personaChat: {'|'.join(assistant_persona)} "
    if chosen_persona:
        assistant_persona = chosen_persona.get("persona", assistant_persona)
        text = f"The user selected a persona: {'|'.join(assistant_persona)} "
        assistant_name = chosen_persona.get("name", assistant_name)
    print(text)

    conv_map[sender_id] = {"assistant_persona": assistant_persona,
                           "user_persona": [],
                           "memory": [],
                           "assistant_name": assistant_name,
                           "messages": [],
                           "last_output_size": 0,
                           "num_memory_access": 0,
                           # "chosen_persona": chosen_persona is not None}
                           "chosen_persona": chosen_persona}

    if get_bool(args.shallow_roleplay):
        ret = f"""
{shallow_desc + sep}

{"The following sentences describe assistant personality and background: " +
 " ".join(conv_map[sender_id]["assistant_persona"]) + sep2 if conv_map[sender_id]["assistant_persona"]
else ""}

{"Complete the following conversation as the assistant with the described character would with a short response: "}

"""
        request_msg = [{"role": "system", "content": ret}]
    else:
        request_msg = [{"role": "request_type", "content": "roleplay_chat"},
                       {"role": "system", "content": system_desc},
                       {"role": "assistant_persona", "content": '||'.join(conv_map[sender_id]["assistant_persona"])},
                       {"role": "assistant_name", "content": conv_map[sender_id]["assistant_name"]}]

    if not chosen_persona:
        request_msg += [{"role": "user", "content": user_utterance}]
    # if the first user_utterance was to define the persona, we don't send it in the request and the
    # assistant will start the conversation

    completion = client.ChatCompletion.create(
        model="vicuna-13b-v1.1",
        messages=request_msg,
        # n=3 if chosen_persona else 1,
        max_tokens=50  # force the bot t0 start with a short message
    )

    bot_first_message = completion.choices[0].message.content
    tokens_usage = completion.choices[0].usage
    conv_map[sender_id]["messages"] = [] if chosen_persona else [{'role': 'user', 'content': user_utterance}]
    conv_map[sender_id]["messages"] += [{'role': 'assistant', 'content': bot_first_message}]
    conv_map[sender_id]["last_output_size"] = tokens_usage["total_tokens"]

    return bot_first_message
    # jsonify({
    #    'persona': ' || '.join(conv_map[sender_id]["assistant_persona"]),
    #    'messages': conv_map[sender_id]["messages"],
    #    'chatbot_utterance': bot_first_message,
    # })


def clean_bot_response(bot_response):
    # TODO: report in a the paper
    bot_response = bot_response.strip()
    string_to_remove = ["En tant que personnage fictif\\s+?,", "un assistant intelligent",
                        "en tant qu['e] \\w+\\s?,",
                        "en tant qu['e](\\s+)?\\S[\\S\\s]*?,"]

    expression_regulieres = re.compile("|".join(string_to_remove), re.IGNORECASE)
    quotes = [r'^"(.*)"$', r"^'(.*)'$", r"^'''(.*)'''$"]
    quotes = re.compile("|".join(quotes), re.IGNORECASE)
    cleaned_bot_response = expression_regulieres.sub("", bot_response)
    cleaned_bot_response = quotes.sub(r'\1', cleaned_bot_response)
    return cleaned_bot_response

def get_bool(toto):
    return str(toto).lower().strip() == "true"

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
        messages=[{"role": "user", "content": get_persona_prompt},
                  {"role": "request_type", "content": "submodule_chat"}],
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
        messages=[{"role": "user", "content": user_persona_gen_prompt},
                  {"role": "request_type", "content": "submodule_chat"}],
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
                                 + seps[1]  # + "\n"
        elif message['role'] == 'user':
            formatted_episode += message["role"] + ": " \
                                 + message["content"] + seps[0]  # + "\n"

    memory_generation_prompt = f"""
Rédige un résumé global de la conversation suivante entre un assistant et un utilisateur en une phrase. 
Le résumé doit comporter des informations pertinentes et ne doit pas commencer par "dans cette conversation".

{formatted_episode}
"""
    completion = client.ChatCompletion.create(
        model="vicuna-13b-v1.1",
        messages=[{"role": "user", "content": memory_generation_prompt},
                  {"role": "request_type", "content": "submodule_chat"}],
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
                        help="Wether or not to run a shallow prompt version of the roleplay.")
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

    sep, sep2 = [" ", "</s>"]
    prompt_length_threshold = args.prompt_length_threshold
    num_turn_threshold = args.num_turn_threshold
    approx_tokens_per_word = args.approx_tokens_per_word

    app.run(host=args.host if args.host else "0.0.0.0",
            port=args.port if args.port else 5008)
