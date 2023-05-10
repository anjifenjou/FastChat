"""
Conversation prompt templates. With add-on to modify the prompt for roleplaying based on personaChat
or user desired persona assignement
"""

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any, Union, Dict, Optional
import json
from json import JSONDecodeError


class SeparatorStyle(Enum):
    """Different separator style."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    BAIZE = auto()
    DOLLY = auto()
    RWKV = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # System prompts
    system: str
    # Two roles
    roles: List[str]
    # All messages
    messages: List[List[str]]
    # Offset of few shot examples
    offset: int
    # Separator
    sep_style: SeparatorStyle
    sep: str

    # New added fields

    # Assistant name in for this conversation
    assistant_name: Optional[str] = None
    # Assistant persona given by the user or randomly selected in personachat
    assistant_persona: Union[str, List[str]] = None
    # User persona detected throughout the conversation by the assistant
    user_persona: Optional[Union[str, List[str]]] = None
    # Caption about an image that user and assistant are supposedly looking at
    # observed_image_captions: Optional[Union[str, List[str]]]
    # Generated memory when history[messages] is truncated
    memory: Optional[Union[List[str], str]] = None
    # result of internet querying about user inputs
    knowledge_response: Optional[str] = None
    # boolean to say if it is roleplay or not
    roleplay: bool = False

    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    # Used for the state in the gradio servers.
    # TODO(lmzheng): refactor this
    conv_id: Any = None
    skip_next: bool = False
    model_name: str = None

    def get_prompt(self):
        # New

        if self.roleplay:  # currently supports Vicuna only
            # print(self.messages)
            assert "vicuna" in self.model_name.lower(), f"Roleplay is only available for Vicuna"
            assert self.sep_style == SeparatorStyle.ADD_COLON_TWO, f"Invalid style: {self.sep_style} for vicuna_v1_1"
            separation = "." + f"{self.sep}"

            self.assistant_persona = [persona.split('.')[0].strip() for persona in self.assistant_persona]
            self.user_persona = [persona.split('.')[0].strip() for persona in self.user_persona]
            # self.observed_image_captions = [caption.split('.')[0].strip() for caption in self.observed_image_captions]

            ret = f"""
            {self.system + self.sep}
            {"You always speak in French." + self.sep}
            {"You act like a regular person. You are not a chabot, your are not an assistant, you are not a computer program"}
            {"You have a name."}
            {"Your name is " + self.assistant_name + "." + self.sep
            if self.assistant_name else ""}
            {"Your personality is: " + separation.join(self.assistant_persona) + self.sep2 if self.assistant_persona
            else ""} 
            {"Remember, you always stay on character. You are the character described above."
             + self.sep2 if self.assistant_persona else ""}  
            {"Don't repeat your personality traits multiple times unless the user asks you to." 
             + self.sep2 if self.assistant_persona else ""} 
            {"You know this about the user you are talking to: " + separation.join(self.user_persona)
             + ". Use it to adapt your conversation to the user" + self.sep2 if self.user_persona else ""}
             
            {"Here is a summary of previous sessions of this conversation to help you remember what has been said: " +
             self.sep.join(self.memory)  if self.memory else ""}
             
             
            {"Complete the following conversation with a short and precise sentence as your character would.  Always speak with new and unique messages that haven't been said in the conversation :"  
            if len(self.messages) > 1 else "Start a conversation in French, empathically as your character would. Write your input only, not the user's response. Do not offer your help, be nice you are talking to a user who just wants to have a chat. You can limit yourself to a greeting:"}

            """
            seps = [self.sep, self.sep2]
            # here we build conversation history
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if role == self.roles[0]:
                        ret += role + ": " + message + seps[0]  # + "\n"
                    elif role == self.roles[1]:
                        ret += role + ": " + message + seps[1]  # + "\n"
                        # ret += self.assistant_name + ": " + message + seps[1]  # + "\n"
                else:
                    if self.knowledge_response:
                        ret += f" Voici des informations supplémentaires récupérées sur Internet" \
                               f" à propos du dernier message de l'utilisateur:" \
                               f"{self.knowledge_response}. Utilise cela pour construire ta réponse. "  # \n"
                    ret += role + ":"
            return ret

        # TODO combine these to make a single code and compatibility with other models
        else:

            if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
                ret = self.system + self.sep
                for role, message in self.messages:
                    if message:
                        ret += role + ": " + message + self.sep
                    else:
                        ret += role + ":"
                return ret
            elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
                seps = [self.sep, self.sep2]
                ret = self.system + seps[0]
                for i, (role, message) in enumerate(self.messages):
                    if message:
                        ret += role + ": " + message + seps[i % 2]
                    else:
                        ret += role + ":"
                return ret
            elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
                ret = self.system
                for role, message in self.messages:
                    if message:
                        ret += role + message + self.sep
                    else:
                        ret += role
                return ret
            elif self.sep_style == SeparatorStyle.BAIZE:
                ret = self.system + "\n"
                for role, message in self.messages:
                    if message:
                        ret += role + message + "\n"
                    else:
                        ret += role
                return ret
            elif self.sep_style == SeparatorStyle.DOLLY:
                seps = [self.sep, self.sep2]
                ret = self.system
                for i, (role, message) in enumerate(self.messages):
                    if message:
                        ret += role + ":\n" + message + seps[i % 2]
                        if i % 2 == 1:
                            ret += "\n\n"
                    else:
                        ret += role + ":\n"
                return ret
            elif self.sep_style == SeparatorStyle.RWKV:
                ret = self.system
                for i, (role, message) in enumerate(self.messages):
                    if message:
                        ret += (
                            role
                            + ": "
                            + message.replace("\r\n", "\n").replace("\n\n", "\n")
                        )
                        ret += "\n\n"
                    else:
                        ret += role + ":"
                return ret
            else:
                raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            conv_id=self.conv_id,
            model_name=self.model_name,
            # New
            assistant_persona=self.assistant_persona,
            assistant_name=self.assistant_name,
            user_persona=self.user_persona,
            memory=self.memory,
            roleplay=self.roleplay,
            knowledge_response=self.knowledge_response
            # observed_image_captions=self.observed_image_captions,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "conv_id": self.conv_id,
            "model_name": self.model_name,
            # New
            "assistant_persona": self.assistant_persona,
            "assistant_name": self.assistant_name,
            "user_persona": self.user_persona,
            "memory": self.memory,
            # "observed_image_captions": self.observed_image_captions
            "roleplay": self.roleplay,
            "knowledge_response": self.knowledge_response
        }


# A template with one conversation example
conv_one_shot = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n### ",
    stop_str="###",
)


# Vicuna v1.1 template
conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
    # New
    assistant_name="Vicuna",
    assistant_persona=[],
    user_persona=[],
    # observed_image_captions=[],
    memory=[],
    roleplay=False,
)

# Koala default template
conv_koala_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)

# Dolly V2 default template
conv_dolly = Conversation(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=("### Instruction", "### Response"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)

# OpenAssistant Pythia default template
conv_oasst = Conversation(
    system="",
    roles=("<|prompter|>", "<|assistant|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="<|endoftext|>",
)

# StableLM Alpha default template
conv_stablelm = Conversation(
    system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
    roles=("<|USER|>", "<|ASSISTANT|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="",
    stop_token_ids=[50278, 50279, 50277, 1, 0],
)

# Baize default template
conv_baize = Conversation(
    system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.",
    roles=("[|Human|]", "[|AI|]"),
    messages=(
        ("[|Human|]", "Hello!"),
        ("[|AI|]", "Hi!"),
    ),
    offset=2,
    sep_style=SeparatorStyle.BAIZE,
    sep="[|Human|]",
    stop_str="[|Human|]",
)

# RWKV-4-Raven default template
conv_rwkv = Conversation(
    system="",
    roles=("Bob", "Alice"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.RWKV,
    sep="",
    stop_str="\n\n",
)

conv_templates = {
    "baize": conv_baize,
    "conv_one_shot": conv_one_shot,
    "dolly": conv_dolly,
    "koala_v1": conv_koala_v1,
    "oasst": conv_oasst,
    "stablelm": conv_stablelm,
    "vicuna_v1.1": conv_vicuna_v1_1,
    "rwkv": conv_rwkv,
}


def get_default_conv_template(model_name):
    model_name = model_name.lower()
    if "vicuna" in model_name or "output" in model_name:
        return conv_vicuna_v1_1
    elif "koala" in model_name:
        return conv_koala_v1
    elif "dolly-v2" in model_name:
        return conv_dolly
    elif "oasst" in model_name and "pythia" in model_name:
        return conv_oasst
    elif "baize" in model_name:
        return conv_baize
    elif "stablelm" in model_name:
        return conv_stablelm
    elif "rwkv-4" in model_name:
        return conv_rwkv
    return conv_one_shot


if __name__ == "__main__":
    conv = conv_templates["vicuna_v1.1"].copy()
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
