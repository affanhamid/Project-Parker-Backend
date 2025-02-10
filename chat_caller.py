import os
from datetime import datetime

from chat_utils import purge_memory, token_counter
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from loguru import logger

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Control parameters
max_total_calls_per_day = int(os.getenv("TOTAL_MODEL_QUOTA"))

# Log-file definition
root_path = os.getenv("ROOT_PATH")
log_path = "logs"
log_file = f"{log_path}/{root_path}_call_log_{{time:YYYY-MM-DD}}.log"
logger.remove()
logger.add(log_file, rotation="1 day", format="{time} {message}", level="INFO")

# Initialize chat model
llm_provider = os.getenv("LLM_PROVIDER")
doc_language = "English"

# Define chat engines

default_model = os.getenv("MODEL_NAME")
chat = ChatOpenAI(temperature=0, model_name=default_model)

# Book-keeping for quota monitoring


def get_daily_calls(log_file):
    """Loop through lines in file and return the number after ' ' on the last line
    This is the cumulative number of calls so far"""
    with open(log_file, "r") as file:
        last_line = None
        for line in file:
            last_line = line
        if last_line:
            return int(last_line.split(" ")[-1])
        else:
            return 0


def check_quota_status():
    try:
        daily_calls_sum = get_daily_calls(
            f"{log_path}/{root_path}_call_log_{datetime.now().strftime('%Y-%m-%d')}.log"
        )
    except FileNotFoundError:
        daily_calls_sum = 0
        logger.remove()
        logger.add(log_file, rotation="1 day", format="{time} {message}", level="INFO")
    return daily_calls_sum


def provide_context_for_question(query, smart_search=False):
    if smart_search == True:
        system = (
            """
        You are an AI that provides assistance in database search. 
        Please translate the user's query to a list of search keywords
        that will be helpful in retrieving documents from a database
        based on similarity.
        The language of the keywords should match the language of the documents: 
        """
            + doc_language
            + """\n
        Answer with a list of keywords.
        """
        )
        query = chat(
            [SystemMessage(content=system), HumanMessage(content=query)]
        ).content
    if os.getenv("DOCS_N") is not None:
        docs = vector_store.similarity_search(query, k=int(os.getenv("DOCS_N")))
    else:
        docs = vector_store.similarity_search(query)
    context = "\n---\n".join(doc.page_content for doc in docs)
    return context


# Read knowledge base
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Avoid warning: https://github.com/huggingface/transformers/issues/5486
)
print("Vector store: " + str(os.getenv("VECTOR_STORE")))
if os.getenv("VECTOR_STORE") is None or os.getenv("VECTOR_STORE") == "faiss":
    print("Using local FAISS.")
    from langchain_community.vectorstores import FAISS

    vector_store = FAISS.load_local(
        os.getenv("CHAT_DATA_FOLDER") + "/faiss_index",
        HuggingFaceInstructEmbeddings(
            cache_folder=os.getenv("MODEL_CACHE"), model_name=model_name
        ),
        allow_dangerous_deserialization=True,
    )


# Admin token check
def check_admin_token(admin_token):
    if (
        admin_token is not None
        and os.getenv("ADMIN_TOKEN") is not None
        and admin_token == os.getenv("ADMIN_TOKEN")
    ):
        model_string = default_model + "-" + admin_token
    else:
        model_string = default_model
    return model_string


instruction_file = open(
    str(os.getenv("CHAT_DATA_FOLDER")) + "/prompt_template.txt", "r"
)
system_instruction_template = instruction_file.read()
print("System instruction template:\n" + system_instruction_template)

# Main chat caller function


def query_gpt_chat(
    query: str,
    history,
    prompt_logging_enabled: bool,
    conversation_id: str,
    admin_token: str = None,
):
    max_tokens = int(os.getenv("MAX_PROMPT_TOKENS"))
    # Check quota status and update model accordingly
    daily_calls_sum = check_quota_status()
    current_model = default_model

    # Search vector store for relevant documents
    context = provide_context_for_question(query)

    # Combine instructions + context to create system instruction for the chat model
    system_instruction = system_instruction_template + context

    # Convert message history to list of message objects
    messages_history = []
    i = 0
    for message in history:
        if i % 2 == 0:
            messages_history.append(HumanMessage(content=message))
        else:
            messages_history.append(AIMessage(content=message))
        i += 1

    # Initialize message list
    messages = [SystemMessage(content=system_instruction)]
    for message in messages_history:
        messages.append(message)
    messages.append(HumanMessage(content=query))

    # Purge memory to save tokens
    # Current implementation is not ideal.
    # Gradio keeps the entire history in memory
    # Therefore, the messages memory is re-purged on every call once token count max_tokens
    # print("Message purge")
    token_count = purge_memory(messages, current_model, max_tokens)
    # print("First message: \n" + str(messages[1].type))
    # print(str(messages))
    # print(token_count)
    if llm_provider != "null":
        results = chat.invoke(messages)
        result_tokens = token_counter([results], default_model)
        print(f"Prompt tokens: {token_count}")
        print(f"Completion tokens: {result_tokens}")
        total_tokens = token_count + result_tokens
        print(f"Total tokens: {total_tokens}")

        # Log statistics
        results_content = results.content
        query_statistics = [token_count, result_tokens, total_tokens, 1]
        if prompt_logging_enabled == True:
            text1 = query.replace("\n", "\\n")
            text2 = results_content.replace("\n", "\\n")
            logged_prompt = f"<{conversation_id}>".join([text1, text2])
        else:
            logged_prompt = "DISABLED"
        model_string = check_admin_token(admin_token)
        # print(model_string)
        query_statistics = (
            model_string
            + ","
            + conversation_id
            + ","
            + logged_prompt
            + ","
            + ",".join(str(i) for i in query_statistics)
            + " "
            + str(daily_calls_sum + 1)
        )
        logger.info(query_statistics)

    else:
        # debug mode:
        results_content = context

    return current_model, results_content


def write_log_removal_request(conversation_id, admin_token):
    daily_calls_sum = check_quota_status()
    model_string = check_admin_token(admin_token)
    logger.info(
        model_string
        + ","
        + conversation_id
        + ","
        + "PROMPT REMOVAL REQUEST"
        + f"<{conversation_id}>"
        + ","
        + ",".join(str(i) for i in [0, 0, 0, 0])
        + " "
        + str(daily_calls_sum)
    )
    return True
