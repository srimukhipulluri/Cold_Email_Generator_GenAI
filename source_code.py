#pip install langchain_groq

# using the api key from chatgroq which uses llama-3.1 model
from langchain_groq import ChatGroq
llm=ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name='llama-3.1-70b-versatile'
)
response=llm.invoke("The first person to land on moon..")
print(response.content)

# pip install langchain_community
# webscraping the given job link to extract the skills,experience,jobdescription
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://jobs.nike.com/job/R-38544?from=job%20search%20funnel")
page_data = loader.load().pop().page_content
print(page_data)

# prompt template for extracting the details in JSON format
from langchain_core.prompts import PromptTemplate

prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scraped text from careers page of a website.
    Your job is to extract the job postings and return in the JSON format containing the following details:'role','experience','skills','description'.
    only return the valid JSON.
    ### valid JSON(NO PREAMBLE):
    """
)
chain_extract = prompt_extract | llm
res=chain_extract.invoke(input={'page_data':page_data})
print(res.content)

# the above output isn't proper json format to convert into json format
from langchain_core.output_parsers import JsonOutputParser
json_parser=JsonOutputParser()
json_res=json_parser.parse(res.content)
print(json_res)

# now the output is in the required json format
type(json_res)

# importing the portfolio dataset
import pandas as pd
df=pd.read_csv('/content/my_portfolio.csv')
df

#pip install chromadb
import chromadb
import uuid
client = chromadb.PersistentClient()
collection=client.get_or_create_collection(name="portfolio")

if not collection.count():
  for _, row in df.iterrows():
    collection.add(documents=row["Techstack"],metadatas={"links": row["Links"]},
    ids=[str(uuid.uuid4())])

# list of links related to the job
links=collection.query(query_texts=["Experience in python","Expertize in React"],n_results=2).get('metadatas')
print(links)

#displays the skills present in the above response given by chatbot
job=json_res
job['skills']


#prompt explaing who you are?(name,your role,your company)
prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}
    ### INSTRUCTION: 
    You are "name here", "your role,company here". NIKE does more than outfit the world’s best athletes. It is a place to explore potential, obliterate boundaries and push out the edges of what can be. The company looks for people who can grow, think, dream and create. Its culture thrives by embracing diversity and rewarding imagination. The brand seeks achievers, leaders and visionaries. At NIKE, Inc. it’s about each person bringing skills and passion to a challenging and constantly evolving game.

NIKE, Inc.’s uncompromising focus on human potential extends to its workforce. The Nike Digital Accessibility team ensures that all athletes, inclusive of disability, have equitable access to Nike’s technology.
Your job is to write a cold email to the client regarding the job mentioned above describing th in fulfilling their needs.

Also add the most relevant ones from the following links to showcase Nike's portfolio: {link_1} Remember you are Srimukhi, ML Engineer at Nike.

Do not provide a preamble.

### EMAIL (NO PREAMBLE):

    """
)
chain_email = prompt_email | llm
# The variable was named link_list in the original code but links in the global variables.
res=chain_email.invoke({"job_description": str(job),"link_1":links})
print(res.content)