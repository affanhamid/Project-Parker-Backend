import os

import tiktoken


def read_course_assets():
    """Read course assets to memory"""
    with open(os.getenv("CHAT_DATA_FOLDER") + "/examples_ui.txt", "r") as file:
        examples = file.readlines()
    examples = [example.strip() for example in examples]

    with open(os.getenv("CHAT_DATA_FOLDER") + "/chat_header.md", "r") as file:
        chat_header = file.read().strip()

    with open(os.getenv("CHAT_DATA_FOLDER") + "/chat_footer.md", "r") as file:
        chat_footer = file.read().strip()

    return examples, chat_header, chat_footer


def purge_memory(messages, model_name, max_tokens: int):
    """Define functions for memory management"""
    token_count = token_counter(messages, model_name)
    # Handle case where system message token count exceeds max_tokens
    system_token_count = token_counter([messages[0]], model_name)
    if system_token_count > max_tokens:
        max_tokens = system_token_count + max_tokens
    if len(messages) > 1:
        while token_count > max_tokens:
            # Print purged message for testing purposes
            # print("Purged the following message:\n" + messages[1].content)
            messages.pop(1)
            token_count = token_counter(messages, model_name)
    return token_count


# PROMPT TOKEN COUNT DOES NOT EXACTLY MATCH OPENAI COUNT
def token_counter(messages, model_name):
    # print("Counting tokens based on: " + current_model)
    encoding = tiktoken.encoding_for_model("gpt-4")
    concatenated_content = "".join([message.content for message in messages])
    token_count = len(encoding.encode(concatenated_content))
    return token_count

