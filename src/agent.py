from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

import pandas as pd
from typing import TypedDict

class GraphInput(TypedDict):
    question: str


class GraphOutput(TypedDict):
    resume: str


class ResumeGraphState(TypedDict):
    question: str
    resume: str
    messages: list

file_url = 'https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset/resolve/main/UpdatedResumeDataSet.csv'

DOMAINS = ['Java Developer', 'Python Developer']

base_prompt = """You are a Resume creator which is capable of creating sample resumes for a given domain.
The domains will be technical like Java Developer, Data Science, DevOps Engineer etc.
You should be able to tailor a strong resume for the provided domain.

Make it relevent to the domain.

Pay attention to the examples below. These are good examples. Generate future resumes in the style of the examples below."""

def get_resume_dataset(state: ResumeGraphState):
  df = pd.read_csv(file_url)
  resume_dataset = []

  for index, row in df.iterrows():
    domain = row["Category"]
    if (domain in DOMAINS):
      resume_dataset.append([domain, row["Resume"]])

  return {
    "question": state["question"],
    "messages": resume_dataset,
  }

def prep_for_fewshot_selection(state: ResumeGraphState):
  resume_dataset = state["messages"]
  messages = [SystemMessage(content = base_prompt)]

  java_dev_examples = 0
  python_dev_examples = 0

  for resume in resume_dataset:
    if java_dev_examples == 5 and python_dev_examples == 5:
      break

    if resume[0] == 'Java Developer' and java_dev_examples < 5:
      human_question = "Create a resume for domain: Java Developer"
      messages.append(HumanMessage(content = human_question))
      messages.append(AIMessage(content = resume[1]))
      java_dev_examples+=1

    if resume[0] == 'Python Developer' and python_dev_examples < 5:
      human_question = "Create a resume for domain: Python Developer"
      messages.append(HumanMessage(content = human_question))
      messages.append(AIMessage(content = resume[1]))
      python_dev_examples+=1
  
  messages.append(HumanMessage(content = state["question"]))

  return {
     "question": state["question"],
     "messages": messages
  }

def create_resume(state: ResumeGraphState):
  messages = state["messages"]
  llm = ChatOpenAI(model_name="gpt-4o")
  resume = llm.invoke(messages)

  return {"resume": resume.content}


resume_workflow = StateGraph(ResumeGraphState, input=GraphInput, output=GraphOutput)
resume_workflow.add_node(get_resume_dataset)
resume_workflow.add_node(create_resume)
resume_workflow.add_node(prep_for_fewshot_selection)

resume_workflow.set_entry_point("get_resume_dataset")
resume_workflow.add_edge("get_resume_dataset", "prep_for_fewshot_selection")
resume_workflow.add_edge("prep_for_fewshot_selection", "create_resume")
resume_workflow.add_edge("create_resume", END)

resume_graph = resume_workflow.compile()