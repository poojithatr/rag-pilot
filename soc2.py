import os
from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load Local models or APIs
mistral = OllamaLLM(model="mistral")

# To load gemini (this API is for free: https://makersuite.google.com/app/apikey)
api_gemini = os.environ.get("GEMINI-API-KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)

# Define the agents
data_collector = Agent(
    role="SOC2 Data Collector",
    goal="""Interact with the end-user to gather SOC2 compliance-related data, and reformat it so that it can be 
    easily assessed against a checklist.""",
    backstory="""You are responsible for engaging with the end-user to collect raw SOC2 compliance-related data. Once the data is received, 
    you organize and reformat it into a structured, checklist-ready format for compliance assessment. Your ability to parse and standardize 
    data ensures that the next steps in the process are seamless.""",
    verbose=True,
    allow_delegation=True,
    # llm=llm
)

compliance_specialist = Agent(
    role="SOC2 Compliance Specialist",
    goal="Develop a comprehensive SOC2 compliance checklist and analyze collected data against the checklist.",
    backstory="""You are an expert in SOC2 compliance and skilled at creating detailed checklists. You review the parsed data provided 
    by the data collector and analyze it against the compliance checklist to determine gaps and areas for improvement.""",
    verbose=True,
    allow_delegation=True,
    # llm=llm
)

report_builder = Agent(
    role="SOC2 Compliance Report Builder",
    goal="Compile a detailed report based on the SOC2 compliance assessment, highlighting strengths and gaps.",
    backstory="""You are responsible for taking the analysis from the compliance specialist and creating a comprehensive, detailed report. 
    The report should highlight areas of compliance, identify gaps, and provide actionable recommendations for achieving SOC2 compliance.""",
    verbose=True,
    allow_delegation=True,
    # llm=llm
)

# Define the tasks
task1 = Task(
    description="""Ask the end-user to upload or provide the SOC2 compliance-related document(s), 
    and reformat it so that it can be easily assessed against a checklist.""",
    agent=data_collector,
    expected_output="A parsed dataset organized for compliance checklist assessment.",
    on_start=lambda: print("Please upload or provide the SOC2 compliance-related document(s) to proceed.")
)

# Execution will prompt for the document when Task 1 begins.


task2 = Task(
    description="""Develop a SOC2 compliance checklist and analyze the collected data against the checklist. Identify areas of compliance and gaps.""",
    agent=compliance_specialist,
    expected_output="A checklist with analysis of the collected data, highlighting compliance and gaps."
)

task3 = Task(
    description="""Build a detailed SOC2 compliance report based on the checklist analysis, including strengths, gaps, and recommendations.""",
    agent=report_builder,
    expected_output="A detailed SOC2 compliance report with strengths, gaps, and actionable recommendations."
)

# Define the crew
crew = Crew(
    agents=[data_collector, compliance_specialist, report_builder],
    tasks=[task1, task2, task3],
    verbose=True,
    process=Process.sequential  # Tasks executed sequentially; the output of one is passed as input to the next.
)

# Execute the workflow
result = crew.kickoff()

print("######################")
print(result)
