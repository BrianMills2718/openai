#attempts to add the counter to the chat function
import sys
from json.decoder import JSONDecodeError
from functools import wraps
import openai
import os
import time
from collections import deque
import json
import pandas as pd
import numpy as np
#import requests
from functools import wraps
import concurrent.futures
from collections import deque
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import HNLoader
from itertools import combinations
from datetime import datetime
import tiktoken
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from threading import Lock  # Make sure you import this
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

def setup_openai():
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai.api_type = "azure"  # Set the API type to 'azure'
    openai.api_base = (
        "https://apigw.rand.org/openai/RAND/inference"  # Set the API base URL
    )
    openai.api_version = "2023-05-15"  # Set the API version
    ##openai.deployment_id="gpt-35-turbo-v0613-base"



##############################################################
# ERROR HANDLING
#max_retries = 100

DEBUG_MODE = True  # Set this to False when not debugging
def debug(message):
    if DEBUG_MODE:
        print("[DEBUG]", message)

# AdaptiveBackoff class definition
class AdaptiveBackoff:
    def __init__(self, initial_wait=23, max_wait=23, history_size=100, check_interval=10):
        self.wait = initial_wait
        self.max_wait = max_wait
        self.history = deque(maxlen=history_size)
        self.check_interval = check_interval
        self.tries = 0

    def record_outcome(self, success):
        self.history.append(success)
        self.tries += 1

    def get_wait_time(self):
        if self.tries % self.check_interval == 0:
            failure_rate = 1 - (sum(self.history) / (1 + len(self.history)))

            if failure_rate > 0.1:
                self.wait *= 1.2
            elif failure_rate < 0.05:
                self.wait *= 0.9

            self.wait = min(self.wait, self.max_wait)
            print
        return self.wait

# Initialize AdaptiveBackoff instance
adaptive_backoff = AdaptiveBackoff()

# Initialize threading lock
lock = Lock()  # Initialize a Lock object

# Handle OpenAI errors decorator
def handle_openai_errors(api_function):
    #adaptive_backoff = adaptive_backoff3.AdaptiveBackoff()  # Initialize AdaptiveBackoff instance
    #adaptive_backoff = AdaptiveBackoff()  # Initialize AdaptiveBackoff instance     #just removed
    @wraps(api_function)
    def wrapper(*args, **kwargs):
        retries = 0
        max_retries = 10000
        final_result = None  # Variable to store the final result

        while retries < max_retries:
            try:
                final_result = api_function(*args, **kwargs)
                adaptive_backoff.record_outcome(True)  # Record a successful API call
                break  # Break the loop if successful
            except openai.error.APIError as e:
                adaptive_backoff.record_outcome(False)  # Record a failed API call
                retries += 1

                status_code = e.json_body.get('statusCode', None)
                
                if status_code == 429:
                    wait_time = adaptive_backoff.get_wait_time()  # Get adaptive wait time
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    print(f"An unknown error occurred with status code {status_code}: {e}")
                    sys.exit(1)  # Exit the program

            except Exception as e:  # Catch all other types of exceptions
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)  # Exit the program

        if retries >= max_retries:
            print("Maximum retries reached. Exiting.")
            sys.exit(1)  # Exit the program after maximum retries

        return final_result  # Return the result after loop finishes

    return wrapper



def handle_openai_errors_concurrent(api_function, max_workers=5):
    @wraps(api_function)
    def wrapper_concurrent(prompts):
        results = []
        counter = 0  # Initialize counter
        estimated_calls = len(prompts)  # Estimate total number of threads
        print(f"Total prompts to process: {estimated_calls}")  # Debug line
        lock = threading.Lock()  # Ensure thread safety
        
        def thread_function(prompt, idx):  # Note the addition of idx here
            nonlocal counter  # To modify counter inside this function
            retries = 0
            max_retries = 1000
            
            while retries < max_retries:
                try:
                    result = api_function(prompt)
                    with lock:
                        adaptive_backoff.record_outcome(True)
                        counter += 1  # Increment the counter
                        print(f"Processing prompt {counter} out of {estimated_calls}")  # Display progress
                    results.append((idx, result))  # Store result as a tuple (index, result)
                    break
                except openai.error.APIError as e:
                    with lock:
                        adaptive_backoff.record_outcome(False)
                    retries += 1

                    status_code = e.json_body.get('statusCode', None)
                    
                    if status_code == 429:
                        with lock:
                            wait_time = adaptive_backoff.get_wait_time()
                        print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
                        time.sleep(wait_time)
                    else:
                        print(f"An unknown error occurred with status code {status_code}: {e}")
                        return
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    return
                
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use enumerate to pass both the prompt and its index
            executor.map(thread_function, prompts, range(len(prompts)))

        # Sort the results by their index and extract only the results
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    return wrapper_concurrent




def handle_openai_errors_concurrent_dict(api_function, max_workers=5):
    @wraps(api_function)
    def wrapper_concurrent(input_sessions_list):
        results = []
        counter = 0
        estimated_calls = len(input_sessions_list)
        print(f"Total chat sessions to process: {len(input_sessions_list)}")

        def thread_function(chat_session, idx):
            retries = 0
            max_retries = 1000
            result = 'Error'
            
            while retries < max_retries:
                try:
                    result = api_function(chat_session)  # Process the entire chat session
                    with lock:
                        adaptive_backoff.record_outcome(True)
                    break
                except openai.error.APIError as e:
                    with lock:
                        adaptive_backoff.record_outcome(False)
                    retries += 1

                    status_code = e.json_body.get('statusCode', None)
                    
                    if status_code == 429:
                        with lock:
                            wait_time = adaptive_backoff.get_wait_time()
                        print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
                        time.sleep(wait_time)
                    else:
                        print(f"An unknown error occurred with status code {status_code}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

            with lock:
                results.append((idx, result))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(thread_function, input_sessions_list, range(len(input_sessions_list)))

        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    return wrapper_concurrent



def old_handle_openai_errors_concurrent_dict(api_function, max_workers=5):

    @wraps(api_function)
    def wrapper_concurrent(input_dicts_list):
        results = []
        counter = 0  # Initialize counter
        estimated_calls = len(input_dicts_list)  # Estimate total number of threads
        print(f"Total prompts to process: {len(input_dicts_list)}")  # Debug line
        
        def thread_function(input_dict, idx):  # Note the addition of idx here
            retries = 0
            max_retries = 1000
            result = 'Error'  # default value
            
            while retries < max_retries:
                try:
                    result = api_function(input_dict['prompt'])  # Extract 'prompt' from input_dict
                    with lock:
                        adaptive_backoff.record_outcome(True)
                except openai.error.APIError as e:
                    with lock:
                        adaptive_backoff.record_outcome(False)
                    retries += 1

                    status_code = e.json_body.get('statusCode', None)
                    
                    if status_code == 429:
                        with lock:
                            wait_time = adaptive_backoff.get_wait_time()
                        print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
                        time.sleep(wait_time)
                    else:
                        print(f"An unknown error occurred with status code {status_code}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

            with lock:
                results.append((idx, result))  # Store result as a tuple (index, result)

                
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use enumerate to pass both the input_dict and its index
            executor.map(thread_function, input_dicts_list, range(len(input_dicts_list)))

        # Sort the results by their index and extract only the results
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    return wrapper_concurrent




#############################################################################
#OPENAI

@handle_openai_errors
def complete(prompt, model="gpt-3.5-turbo-16k"):
        setup_openai()  # Uncomment this line when you have the function
        print("ENTERING COMPLETE")
        #print("INPUT PROMPT: ", prompt, "\n")
        messages= [
            {
                "role": "system",
                "content": "You are a helpful assistant",
            },
            {
                "role": "user",
                "content": f"{prompt}",
            },
        ]
        #print("FULL INPUT MESSAGES:", messages")  # Uncomment this line when you have the function
        # Assuming you've imported OpenAI API above
        chat_completion = openai.ChatCompletion.create(
            deployment_id="gpt-35-turbo-v0301-base",
            model="gpt-3.5-turbo-16k",
            messages=messages,
            headers={"Ocp-Apim-Subscription-Key": os.environ.get("OPENAI_API_KEY")},
        )
        #debug(chat_completion.choices[0].message.content)  # Uncomment this line when you have the function
        print("INPUT PROMPT: ", prompt, "\n")
        print("OUTPUT: ", chat_completion.choices[0].message.content) 
        print("EXITING CHAT \n\n")
        return chat_completion.choices[0].message.content

#COMPLETE4
@handle_openai_errors
def complete4(prompt):
    return complete(prompt, model="gpt-4-32k-0613")


#CHAT3.5

@handle_openai_errors
def chat(messages):
    setup_openai()
    print("ENTERING CHAT")
    print("INPUT MESSAGE: ", messages, "\n")
    chat_completion = openai.ChatCompletion.create(
        deployment_id="gpt-35-turbo-v0301-base",
        model="gpt-3.5-turbo-16k",
        messages=messages,
        headers={"Ocp-Apim-Subscription-Key": os.environ.get("OPENAI_API_KEY")},
    )
    print("OUTPUT: ", chat_completion.choices[0].message.content)
    print("EXITING CHAT \n\n")
    return chat_completion.choices[0].message.content
    
#CHAT4
@handle_openai_errors
def chat4(messages):
    return chat(messages, model="gpt-4-32k-0613")

#############################################################################
#CONCURRENT 

con_complete= handle_openai_errors_concurrent(complete)
con_complete4= handle_openai_errors_concurrent(complete4)
#con_chat= handle_openai_errors_concurrent_dict(chat)
con_chat=handle_openai_errors_concurrent(chat)
#this doesn't work, i think because the handleopenaierrorsconcurrent can only take lsits and chat takes dictionaries
con_chat4= handle_openai_errors_concurrent_dict(chat4)

################################
#Token Counter for text GPT3.5
def token_counter(message, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages, i.e. the thing use by OpenAI CHat Completions"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Key Error, switching to cl100k_base for embedding intstead of OpenAI (or at least I think thats what this does - Mills)")
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens=len(encoding.encode(message))
    return num_tokens
#GPT4
def token_counter4(messages):
    return complete(messages, model="gpt-4-32k-0613")
#GPT3.5 Concurrent
con_token_counter= handle_openai_errors_concurrent(token_counter)
#
#GPT4 Concurrent
con_token_counter4= handle_openai_errors_concurrent(token_counter4)



#Token Counter for Chat Messages GPT3.5
def chat_token_counter(messages, model="gpt-3.5-turbo-0613"):
  """Returns the number of tokens used by a list of messages, i.e. the thing use by OpenAI CHat Completions"""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      print("Key Error, switching to cl100k_base for embedding intstead of OpenAI")
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
#
#GPT4
def chat_token_counter4(messages):
    return complete(messages, model="gpt-4-32k-0613")
#GPT3.5 Concurrent
chat_con_token_counter= handle_openai_errors_concurrent(token_counter)
#
#GPT4 Concurrent
chat_con_token_counter4= handle_openai_errors_concurrent(token_counter4)


#TEXT SPLITTING

def split(long_text, size=20000, overlap=0):
    # Initialize an empty list to store the smaller chunks of text
    texts = []
    
    # Initialize the start and end index for slicing the string
    start = 0
    end = size
    
    # Counter to track which text number we're currently on
    counter = 1
    if len(long_text) <= size:
        return [long_text]

    # Loop to create smaller chunks
    while start < len(long_text):
        # Initialize the split index to the end index
        split_index = end
        
        # Look for two new lines within the overlap window
        two_new_lines_index = long_text[start:end + overlap].rfind('\n\n')
        if two_new_lines_index != -1:
            split_index = start + two_new_lines_index + 2
            
        # If two new lines not found, look for a single new line
        elif long_text[start:end + overlap].rfind('\n') != -1:
            split_index = start + long_text[start:end + overlap].rfind('\n') + 1
            
        # If a single new line is also not found, look for a period
        elif long_text[start:end + overlap].rfind('.') != -1:
            split_index = start + long_text[start:end + overlap].rfind('.') + 1
        
        # If none of the above are found, split at the designated size
        else:
            split_index = end

        # Check if the split_index exceeds the length of the long_text
        if split_index > len(long_text):
            split_index = len(long_text)
            
        # Append the chunk to the list
        texts.append(long_text[start:split_index])

        # Print the current text number and total count
        print(f"Splitting text {counter} out of an estimated {int(len(long_text) / size)}")
        
        # Update start index for the next iteration
        start = split_index
    
        # Update the end index for the next iteration
        end = start + size

        # Increment the counter
        counter += 1
    
    print("You have: ", len(texts), "texts \n")
    return texts


def split_on_word(long_text, size=10000):
    # Initialize an empty list to store the smaller chunks of text
    texts = []
    
    # Initialize the start index for slicing the string
    start = 0
    
    # Loop to create smaller chunks
    while start < len(long_text):
        # Calculate the end index for the current chunk
        end = min(start + size, len(long_text))
        
        # If we're not at the end of the text, find the nearest space to split on
        if end < len(long_text):
            while end > start and long_text[end] != ' ':
                end -= 1
        
        # Append the chunk to the list
        texts.append(long_text[start:end])
        
        # Update start index for the next iteration
        start = end
    
    return texts

def split_on_newline(long_text, size=10000):
    # Initialize an empty list to store the smaller chunks of text
    texts = []
    
    # Initialize the start and end index for slicing the string
    start = 0
    end = size
    
    # Loop to create smaller chunks
    while start < len(long_text):
        # If the remaining text is smaller than the chunk size, append it and break
        if end >= len(long_text):
            texts.append(long_text[start:])
            break
        
        # Look backwards from the end index for the nearest newline character
        newline_index = long_text.rfind('\n', start, end)
        
        # If a newline is found, adjust the end index to split at the newline
        if newline_index != -1:
            end = newline_index + 1
            
        # Append the chunk to the list
        texts.append(long_text[start:end])
        
        # Update start and end indexes for the next iteration
        start = end
        end = start + size
    
    return texts



def candeletesplit(long_text, size=4000, overlap=500):
    # Initialize an empty list to store the smaller chunks of text
    texts = []
    
    # Initialize the start and end index for slicing the string
    start = 0
    end = size
    
    # Loop to create smaller chunks
    while start < len(long_text):
        # Initialize the split index to the end index
        split_index = end
        
        # Look for two new lines within the overlap window
        two_new_lines_index = long_text[start:end + overlap].rfind('\n\n')
        if two_new_lines_index != -1:
            split_index = start + two_new_lines_index + 2
            
        # If two new lines not found, look for a single new line
        elif long_text[start:end + overlap].rfind('\n') != -1:
            split_index = start + long_text[start:end + overlap].rfind('\n') + 1
            
        # If a single new line is also not found, look for a period
        elif long_text[start:end + overlap].rfind('.') != -1:
            split_index = start + long_text[start:end + overlap].rfind('.') + 1
        
        # If none of the above are found, split at the designated size
        else:
            split_index = end

        # Check if the split_index exceeds the length of the long_text
        if split_index > len(long_text):
            split_index = len(long_text)
            
        # Append the chunk to the list
        texts.append(long_text[start:split_index])
        
        # Update start index for the next iteration
        start = split_index
    
        # Update the end index for the next iteration
        end = start + size
    
    print("You have: ", len(texts), "texts \n")
    return texts


def old_split(long_text,size= 4000 , overlap=500):   
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=size,
    chunk_overlap= overlap,
    )
    texts = text_splitter.create_documents([long_text])
    print(f"You have {len(texts)} documents")

    print("Preview:")
    print(texts[0].page_content, "\n")
    print(texts[1].page_content)
    return(texts)




############################################
#EMBEDDING
@handle_openai_errors
def embed(input_text):
    debug("entering embedding function \n")
    setup_openai()
    embeddings_response = openai.Embedding.create(
        input=input_text,
        deployment_id="text-embedding-ada-002-v2-base",
        headers={"Ocp-Apim-Subscription-Key": os.environ.get("OPENAI_API_KEY")}
    )
    debug(embeddings_response['data'][0]['embedding'])
    return embeddings_response['data'][0]['embedding']
#
con_embed= handle_openai_errors_concurrent(embed)

def similarity(vectors1, vectors2=None):
    # Check if vectors2 is None, meaning we need to compare vectors within vectors1
    single_list_mode = vectors2 is None
    
    # Detect the input types for descriptive messaging
    type_desc1 = "list of vectors" if isinstance(vectors1[0], list) else "single vector"
    
    if not single_list_mode:
        type_desc2 = "list of vectors" if isinstance(vectors2[0], list) else "single vector"
        print(f"You fed in a {type_desc1} to compare it to a {type_desc2}.")
    else:
        print(f"You fed in a {type_desc1} and are comparing each vector in the list to each other.")
    
    # Ensure vectors1 is a list of vectors. If it's a single vector, convert it to a list.
    if not isinstance(vectors1[0], list):
        vectors1 = [vectors1]
    
    if not single_list_mode:
        # Ensure vectors2 is a list of vectors. If it's a single vector, convert it to a list.
        if not isinstance(vectors2[0], list):
            vectors2 = [vectors2]
    else:
        vectors2 = vectors1  # In single list mode, compare vectors within the same list
    
    similarities = []
    
    for vec1 in vectors1:
        vec1 = np.array(vec1).reshape(1, -1)
        for vec2 in vectors2:
            vec2 = np.array(vec2).reshape(1, -1)
            
            # Skip comparing the same vectors in single list mode
            if single_list_mode and np.array_equal(vec1, vec2):
                continue
                
            similarity = cosine_similarity(vec1, vec2)[0][0]
            similarities.append((vec1.tolist()[0], vec2.tolist()[0], similarity))
    
    print(f"Output: A list of tuples. Each tuple contains two lists (vectors) and a float (similarity score).")

    return similarities


def similarity_table(similarity_data):
    # Initialize an empty DataFrame
    df = pd.DataFrame()

    for idx1, idx2, sim in similarity_data:
        # Use index positions as labels
        idx1_label = f"Vec_{similarity_data.index((idx1, idx2, sim)) + 1}_1"
        idx2_label = f"Vec_{similarity_data.index((idx1, idx2, sim)) + 1}_2"

        # Populate the DataFrame
        df.loc[idx1_label, idx2_label] = sim
        df.loc[idx2_label, idx1_label] = sim  # Similarity is commutative

    # Filling diagonal with 1s as any vector is perfectly similar to itself
    for idx in df.index:
        df.loc[idx, idx] = 1.0

    # Filling NaNs with zeros
    df.fillna(0, inplace=True)

    print("Similarity Table:")
    display(df)
    return df

def build_similarity_matrix(embedded_vectors, similarity_data):
    num_texts = len(embedded_vectors)
    matrix = np.zeros((num_texts, num_texts))
    
    for vec1, vec2, sim in similarity_data:
        i = [j for j, emb in enumerate(embedded_vectors) if np.array_equal(emb, vec1)][0]
        j = [j for j, emb in enumerate(embedded_vectors) if np.array_equal(emb, vec2)][0]
        matrix[i, j] = sim
        matrix[j, i] = sim  # Assuming similarity is symmetric
    
    return matrix


def build_graph_from_list(texts):
    # Step 2: Generate embeddings for these texts
    embedded_vectors = [con_embed(text) for text in texts]
    
    # Step 3: Calculate similarity
    similarity_data = similarity(embedded_vectors)
    
    # Output similarity matrix
    sim_matrix = build_similarity_matrix(embedded_vectors, similarity_data)
    print("Similarity Matrix:")
    print(sim_matrix)

    # Step 4: Build the graph
    G = nx.Graph()

    # Add nodes, using the texts as labels
    for i, text in enumerate(texts):
        G.add_node(i, label=text)

    # Add edges with weights and labels
    for vec1, vec2, sim in similarity_data:
        i = [j for j, emb in enumerate(embedded_vectors) if np.array_equal(emb, vec1)][0]
        j = [j for j, emb in enumerate(embedded_vectors) if np.array_equal(emb, vec2)][0]
        G.add_edge(i, j, weight=sim, label=f"{sim:.2f}")

    pos = nx.spring_layout(G)  # positions for all nodes
    
    # Extract labels from nodes
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    
    nx.draw(G, pos, labels=labels, with_labels=True)
    
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()

def build_graph_from_list_old(texts):

    # Step 2: Generate embeddings for these texts
    embedded_vectors = [con_embed(text) for text in texts]  # Using your existing 'con_embed' function
    
    # Step 3: Calculate similarity
    similarity_data = similarity(embedded_vectors)
    
    # Step 4: Build the graph
    G = nx.Graph()

    # Add nodes, using the texts as labels
    for i, text in enumerate(texts):
        G.add_node(i, label=text)

    # Add edges with weights and labels
    for vec1, vec2, sim in similarity_data:
        node1 = [i for i, emb in enumerate(embedded_vectors) if emb == vec1][0]
        node2 = [i for i, emb in enumerate(embedded_vectors) if emb == vec2][0]
        G.add_edge(node1, node2, weight=sim, label=f"{sim:.2f}")

    pos = nx.spring_layout(G)  # positions for all nodes
    
    # Extract labels from nodes
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    
    nx.draw(G, pos, labels=labels, with_labels=True)
    
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()


def build_graph2(original_text):
    # Step 1: Split the text into smaller parts
    texts = split(original_text)
    
    # Step 2: Generate embeddings for these texts
    embedded_vectors = [con_embed(text) for text in texts]  # Using your existing 'con_embed' function
    
    # Step 3: Calculate similarity
    similarity_data = similarity(embedded_vectors)
    
    # Step 4: Build the graph
    G = nx.Graph()

    # Add nodes, using the texts as labels
    for i, text in enumerate(texts):
        G.add_node(i, label=text)

    # Add edges with weights and labels
    for vec1, vec2, sim in similarity_data:
        node1 = [i for i, emb in enumerate(embedded_vectors) if emb == vec1][0]
        node2 = [i for i, emb in enumerate(embedded_vectors) if emb == vec2][0]
        G.add_edge(node1, node2, weight=sim, label=f"{sim:.2f}")

    pos = nx.spring_layout(G)  # positions for all nodes
    
    # Extract labels from nodes
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    
    nx.draw(G, pos, labels=labels, with_labels=True)
    
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()




def newer_build_graph(original_text):
    # Step 1: Split the text into smaller parts
    texts = split(original_text)
    
    print(f"Debug: texts = {texts}")  # Debugging line
    
    # Step 2: Generate embeddings for these texts
    embedded_vectors = [embed(text) for text in texts]
    
    print(f"Debug: embedded_vectors = {embedded_vectors}")  # Debugging line
    
    # Step 3: Calculate similarity
    similarity_data = similarity(embedded_vectors)
    
    print(f"Debug: similarity_data = {similarity_data}")  # Debugging line
    
    # Step 4: Build the graph
    G = nx.Graph()

    # Add nodes, using the original_texts as labels
    for i, text in enumerate(texts):
        G.add_node(i, label=text)

    print(f"Debug: G.nodes = {G.nodes(data=True)}")  # Debugging line
    
    # Add edges with weights and labels
    for i, data in enumerate(similarity_data):
        vec1, vec2, sim = data
        G.add_edge(i, len(texts) + i, weight=sim, label=f"{sim:.2f}")

    pos = nx.spring_layout(G)  # positions for all nodes
    
    # Extract labels from nodes
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    
    nx.draw(G, pos, labels=labels, with_labels=True)

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()



def old_build_graph(similarity_data):
    G = nx.Graph()

    # Add nodes, using indexes as labels
    for i in range(len(similarity_data)):
        G.add_node(i)

    # Add edges with weights and labels
    for i, data in enumerate(similarity_data):
        vec1, vec2, sim = data
        G.add_edge(i, len(similarity_data) + i, weight=sim, label=f"{sim:.2f}")

    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True)

    labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()

    return G




    



################################

#IMAGE
@handle_openai_errors
def image(caption, n=1):
#def image(caption, number=1):

    setup_openai()
    dalle_response = requests.post(
        'https://apigw.rand.org/openai/RAND/images/generations:submit?api-version=2023-06-01-preview',
        headers = { "Ocp-Apim-Subscription-Key": os.environ.get("OPENAI_API_KEY"), "Content-Type": "application/json" }
,
        json = {
            'prompt': f'{caption}',
            #'n': 1,
            'n': n,
            'resolution': '1024X1024'
        }
    )
    print(caption)
    operation_location = dalle_response.headers['Operation-Location']

    status = ""

    while status != "succeeded":
        status_response = requests.get(operation_location, 
        headers = { "Ocp-Apim-Subscription-Key": os.environ.get("OPENAI_API_KEY"), "Content-Type": "application/json" })
        status = status_response.json()['status']
        if status in ['notRunning', 'running']:
            time.sleep(int(status_response.headers['Retry-After']))

    image_url = status_response.json()['result']['data'][0]['url']




    # Download the image and create a PIL Image object
    image_response = requests.get(image_url)
    img = Image.open(BytesIO(image_response.content))

    # Display the image using matplotlib
    plt.imshow(img)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


    debug(image_url)
    return image_url


con_image= handle_openai_errors_concurrent(image)

##################################################
#CHAIN REDUCE SECTION


def reduce(long_text):
    texts=split(long_text)
    summary_list=[]
    for text in texts:
        messages = [
                {
                    "role": "system",
                    "content": f"You are an expert summarizer of texts. You will receive a piece of text which may have been cut from a random point in a document. Your job is to do your best to summarize whatever text your receive while trying to keep the formatting and style of your output as close as possible to the input text. Try to keep as much of the specific terminology and phrases as possible. Leave as much unchanged as possible. You want to keep as much of the details as possible while reducing the size of the original text. Reduce the size by of the document by roughly about half. ",
                },
                {
                    "role": "user",
                    "content": f"{text}",
                },
        ]
        summary = chat(messages)
        summary_list.append(summary)
        return summary_list
chat_con_token_counter4= handle_openai_errors_concurrent(token_counter4)


def map_reduce(long_text, summary_list=None):
    if summary_list is None:
        summary_list = []
        
    # Call your reduce function to get the initial list of summaries.
    reduced_summaries = reduce(long_text)
    
    for summary in reduced_summaries:
        # Count the number of tokens in the text.
        token_count = token_counter(summary)
        
        # If the token count is greater than 4000, split the text.
        if token_count > 10000:
            sub_texts = split(summary)  # Assuming split returns list of texts with a `.page_content` attribute
            
            # Loop through each split text and recursively call map_reduce.
            for sub_text in sub_texts:
                map_reduce(sub_text.page_content, summary_list)
                
        else:
            # If the summary has less than 4000 tokens, append it to the summary list.
            summary_list.append(summary)
            
    # Join all the summaries into a single string separated by new lines.
    final_summary = '\n'.join(summary_list)
    
    return final_summary
    







"""
###############################
#CHAINS

def api_call(prompt):
    # Simulate an API call
    # In a real-world scenario, you would use OpenAI's API here
    print(f"API called with prompt: {prompt}")
    time.sleep(1)
    return prompt[::-1]  # Reverse the string for demonstration

def chain_prompts(api_function, initial_prompt, num_chains):
    current_prompt = initial_prompt
    for i in range(num_chains):
        # Calling the API function and storing its output
        output = api_function(current_prompt)
        
        # Making the output of the current API call as the input for the next one
        current_prompt = output
    
    return current_prompt

# Usage
initial_prompt = "hello"
num_chains = 3
final_output = chain_prompts(api_call, initial_prompt, num_chains)
print("Final output:", final_output)

"""