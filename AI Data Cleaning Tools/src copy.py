import os
import sys
from dotenv import load_dotenv
load_dotenv()
import time
sys.path.append('../')
sys.path.append('/')
from copy import deepcopy
import openai



account_key = os.environ['PERSONAL_KEY']  # Your Storage Account Key


# Set OpenAI API Key
openai.api_key = account_key

def return_response(message):
    client = openai.OpenAI(api_key=account_key)
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are a professional data analyst."},
            {"role": "user", "content": message}
        ],
        temperature=0.0,  # Controls randomness (0 = deterministic, 1 = very creative)
        max_tokens=5000  # Limit response length
    )
    return response

def return_response_o1(message):
    client = openai.OpenAI(api_key=account_key)
    response = client.chat.completions.create(
        model="o1-preview", 
        messages=[
            {"role": "user", "content": message}
        ],
        max_completion_tokens=32768,  # Limit response length,
    )
    return response


import pandas as pd
csv_registry_df = pd.read_csv("../data/csv_registry_V2.csv")
csv_registry = csv_registry_df.to_dict(orient='records')

csv_registry = [el for el in csv_registry if el.get('clean_csv_path') is not None]

batch_size = 10

def process_batch(batch, research_question):
    # Your processing logic here
    filename_description_dict = str({el['clean_csv_path']: el['full_description']
                                 for el in batch})
    
    research_query = f"""{research_question}

Based on descriptions of the datasets provided, select and output the file_names of the datasets that are HIGHLY RELEVANT to the research task at hand as a list of string file names.

If none seems relevant, output empty list like [] with NO ADDITIONAL COMMENTS.

Do NOT return any JSON markdown in the answer, return a simple string with square brackets and comma separated file names like so: ['file_name1', 'file_name2', 'file_name3']

Descriptions of the datasets (as file_name - to - description map): {filename_description_dict}

Output:

"""
    response = return_response(research_query)
    return(response)


# Iterate over csv_registry in batches of 5
all_related_datasets = []

def dataset_collation_prompt(dataset_list, research_question): 
    full_dataset_descriptions_lst = []
    i = 0
    for el in csv_registry:
        if el['clean_csv_path'] in dataset_list:
            el_copy = deepcopy(el)
            el_copy.pop('file_name')
            el_copy.pop('file_path')
            el_copy.pop('full_description')

            i += 1
            full_dataset_descriptions_lst.append(f"dataset {str(i)} - {str(el_copy)}")

    full_dataset_descriptions = " ".join(full_dataset_descriptions_lst)
    
    final_prompt = f""" You are a senior data analyst specializing in data cleaning and transformation. 
    Based on the research question, csv decriptions and a list of csv files, return a data modelling plan that outlines the following:
    * Out of all the datasets provided, selects the relevant datasets based on the list of csv files, and if multiple suitable similar csvs are present. Specify which ones are using by the clean_csv_path.
    * Performs data cleaning operations to ensure the datasets are in a usable format.
    * If multiple tables in CSV, split tables and process separately.
    * When aggregating by quarter, ensure that the quarter date is the LAST date in the quarter.
    * Standartizes column and row naming conventions of the datasets with short, distinct names. Checks that the number of columns matches he assigned list of standard column names to avoid errors.
    * Performs data joins via pandas merge and concat functions to unify the datasets into one dataframe.

    It should be in plain text with some pandas functions suggested (no code examples needed).

    In addition, provide a short description of each column in the resulting dataframe using the revised column names.

    The research question is: {research_question}
    
    The datasets are as follows: {full_dataset_descriptions}

    """

    response = return_response_o1(final_prompt)
    response_text = response.choices[0].message.content

    return(response_text)

    
    

def qa(research_question):

    all_related_datasets = []

    for i in range(0, len(csv_registry), batch_size):
        batch = csv_registry[i:i + batch_size]
        related_datasets = process_batch(batch, research_question)
        time.sleep(2)
        print(related_datasets.choices[0].message.content)
        related_datasets = eval(related_datasets.choices[0].message.content)
        all_related_datasets.extend(related_datasets)
    response = dataset_collation_prompt(all_related_datasets, research_question)
    return response