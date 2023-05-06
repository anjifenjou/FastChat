import os
from fastchat import client
import json
from pprint import pformat
from flask import Flask, request, jsonify
from argparse import ArgumentParser
from datasets import load_dataset
import random
from nltk.tokenize import RegexpTokenizer
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

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

    else:
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
            request_messages += {"role": "user_persona", "content": '||'.join(user_persona)}

        ###############################################################################################################
        #                               ACCESSING MEMORY IF CONDITION REACHED (modify history then)
        ###############################################################################################################
        approximate_input_tokens = approx_tokens_per_word * words_count(user_utterance)
        approx_new_prompt_length = conv_map[sender_id]["last_output_size"] + approximate_input_tokens
        # + new_user_persona_length

        if approx_new_prompt_length >= prompt_length_threshold:  # access memory when reaching condition
            # TODO actually we should compare approx_new_prompt_length + max_new_tokens to max context size
            # prompt_length_threshold  linked to max_new_tokens and max_possible_context of the considered model (LLaMA)
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
        )

        tokens_usage = completion.choices[0].usage  # usage is now in choices
        bot_response = completion.choices[0].message.content

        ################################################################################################################
        #                                            UPDATE CONVERSATION HISTORY
        ################################################################################################################
        conv_map[sender_id]["messages"] += [{'role': 'user', 'content': user_utterance},
                                            {'role': 'assistant', 'content': bot_response}]
        conv_map[sender_id]["last_output_size"] = tokens_usage["total_tokens"]
        history = conv_map[sender_id]["messages"]

        ################################################################################################################
        #                                           USER PERSONA BUILDING MODULE
        ################################################################################################################
        if history and len(history) % (num_turn_threshold * 2) == 0:
            # build user persona in this episode:
            # stop = len(history)  # // (num_turn_threshold * 2)
            # start = len(history) - num_turn_threshold * 2
            user_utterances = [message for message in history[- num_turn_threshold * 2:] if message["role"] == 'user']
            conv_map[sender_id]["user_persona"].append(generate_user_persona(user_utterances))

        ################################################################################################################
        #                                               MEMORY BUILDING MODULE
        ################################################################################################################
        if history and len(history) % num_turn_threshold == 0:
            if conv_map[sender_id]["memory"]:
                start = conv_map[sender_id]["memory"][-1]["stop"]
            else:
                start = 0
            stop = start + num_turn_threshold
            conv_map[sender_id]["memory"].append(generate_memory(history[start: stop],
                                                                 start=start, stop=stop))

    print(bot_response)
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
    chosen_persona = get_desired_persona(user_utterance=user_utterance)  # The case of user desired persona.
    text = f"Persona is randomly assigned from personaChat: {'|'.join(assistant_persona)} "
    if chosen_persona:

        assistant_persona = chosen_persona.get("persona", assistant_persona)
        text = f"The user selected a persona: {'|'.join(assistant_persona)} "
        assistant_name = chosen_persona.get("name", assistant_name)
    print(text)

    conv_map[sender_id] = {"assistant_persona": assistant_persona,
                           "user_persona": [],
                           "memory": [],
                           "name": assistant_name,
                           "messages": [],
                           "last_output_size": 0,
                           "num_memory_access": 0}

    # Send a request to the API to make the bot start the conversation
    request_msg = [{"role": "request_type", "content": "roleplay_chat"},
                   {"role": "system", "content": system_desc},
                   {"role": "assistant_persona", "content": '||'.join(conv_map[sender_id]["assistant_persona"])},
                   {"role": "assistant_name", "content": conv_map[sender_id]["assistant_name"]},
                   ] + [] if chosen_persona else [{"role": "user", "content": user_utterance}]
    # if the first user_utterance was to define the persona, we don't send it in the request and the
    # assistant will start the conversation

    completion = client.ChatCompletion.create(
        model="vicuna-13b-v1.1",
        messages=request_msg,
        n=3 if chosen_persona else 1,
        max_tokens=50  # force the bot t0 start with a short message
    )

    bot_first_message = completion.choices[0].message.content
    tokens_usage = completion.choices[0].usage
    conv_map[sender_id]["messages"] = [] if chosen_persona else [{'role': 'assistant', 'content': bot_first_message}]
    conv_map[sender_id]["messages"] += [{'role': 'assistant', 'content': bot_first_message}]
    conv_map[sender_id]["last_output_size"] = tokens_usage["total_tokens"]

    return bot_first_message
    # jsonify({
    #    'persona': ' || '.join(conv_map[sender_id]["assistant_persona"]),
    #    'messages': conv_map[sender_id]["messages"],
    #    'chatbot_utterance': bot_first_message,
    # })


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
    get_persona_prompt = f"""
    Dans le texte ci-dessous, vérifie si des éléments par rapport à un nom ou des traits de personnalité sont donnés. 
    Si c'est le cas retourne les informations au format json avec les champs suivants: name (si applicable) et  persona (une liste de chaines des caractères extraites du texte).
    Si ce n'est pas le cas retourne "None".

    Texte:  
        "{user_utterance}"
    """  # positivity in the prompt

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
        chosen_persona = json.loads(persona_response)
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
        formatted_episode += message["role"] + ": " \
                             + message["content"] \
                             + seps[1] if message['role'] == 'assistant' else seps[0] + "\n"

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
    request_memory_content = [content for content in conv_map[sender_id]["memory"][:memory_index]]
    new_history_start = conv_map[sender_id]["memory"][:memory_index][-1]["stop"]
    # new_history_start -= 2   # to make sure we have at least
    history = conv_map[sender_id]["messages"][new_history_start:]
    if len(history) < 2:
        history = conv_map[sender_id]["messages"][new_history_start - 2:]
    return request_memory_content, history


def get_search_decision(user_utterance):
    # TODO add a designed search decision prompt later
    search_decision_prompt = f"""
    """
    search_decison = False
    return search_decison


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
    parser.add_argument("--host", type=str, default=None,
                        help="Address of the flask_server host")
    parser.add_argument("--port", type=str, default=None,
                        help="port of the flask_server ")
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
    system_desc = "A chat between a curious user and an artificial intelligence assistant. " \
                  "The assistant is engaging, empathetic, and role plays as the character described below. " \
                  "The assistant gives helpful, detailed and polite an answers to user's questions " \
                  "and the assistant feels free to ask questions to the user."
    prompt_length_threshold = args.prompt_length_threshold
    num_turn_threshold = args.num_turn_threshold
    approx_tokens_per_word = args.approx_tokens_per_word

    app.run(host=args.host if args.host else "0.0.0.0",
            port=args.port if args.port else 5008)
