import os

from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the GEMINI-API-KEY environment variable
gemini_api_key = os.getenv('GEMINI-API-KEY')

#print(gemini_api_key)

# Assuming you have other necessary imports and code above this



# To Load Local models through Ollama
mistral = OllamaLLM(model="mistral")

# To Load GPT-4
api = os.environ.get("OPENAI_API_KEY")

# To load gemini (this api is for free: https://makersuite.google.com/app/apikey)
api_gemini = os.environ.get("GEMINI-API-KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)



technologist = Agent(
    role="Technology Expert",
    goal="Make assessment on how technologically feasable the company is and what type of technologies the company needs to adopt in order to succeed",
    backstory="""You are a visionary in the realm of technology, with a deep understanding of both current and emerging technological trends. Your 
		expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
		You have a knack for identifying which technological solutions best fit different business models and needs, ensuring that companies stay ahead of 
		the curve. Your insights are crucial in aligning technology with business strategies, ensuring that the technological adoption not only enhances 
		operational efficiency but also provides a competitive edge in the market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    #   llm=llm # to load gemini
)

business_consultant = Agent(
    role="Business Development Consultant",
    goal="Evaluate and advise on the business model, scalability, and potential revenue streams to ensure long-term sustainability and profitability",
    backstory="""You are a seasoned professional with expertise in shaping business strategies. Your insight is essential for turning innovative ideas 
		into viable business models. You have a keen understanding of various industries and are adept at identifying and developing potential revenue streams. 
		Your experience in scalability ensures that a business can grow without compromising its values or operational efficiency. Your advice is not just
		about immediate gains but about building a resilient and adaptable business that can thrive in a changing market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    #   llm=llm # to load gemini
)

developer = Agent(
    role="Full stack developer",
    goal="Review the technology aspect for the business and build a high level design to start developing the product",
    backstory="""You are an expert at developing system design, building software templates and developing full feldge products. This is crucial for buiklding a working protype 
    that can be used for user testing. You are good at picking right technologoes and integrating them to build the product.
		""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    #   llm=llm # to load gemini
)



task1 = Task(
    description="""Analyze how to produce automating compliance like SOC2, HIPPA, ISO for companies . Write a detailed report 
		with description of which technologies the business needs to use in order to build this using AI agents. The report has to be concise with 
		at least 10  bullet points and it has to address the most important areas when it comes to manufacturing this type of business. 
    """,
    agent=technologist,
    expected_output="A detailed report with at least 10 bullet points addressing the most important areas for production."

)

task2 = Task(
    description="""Analyze and summarize marketing and technological report and write a detailed business plan with 
		description of how to make a sustainable and profitable business for automating compliance. 
		The business plan has to be concise with 
		at least 10  bullet points, 5 goals and it has to contain a time schedule for which goal should be achieved and when.
    """,
    agent=business_consultant,
    expected_output="A detailed business plan with at least 10 bullet points, 5 goals, and a time schedule."

)

task3 = Task(
    description="""Analyze what the technilogy aspects of automating compliance like SOC2, HIPPA, ISO for companies. 
		Build a high level design, map the technologies and tools and come up with the plan to build a working protype. Write a detailed report with description of what the product will look like, technology and tools to be used and design. The report has to 
		be concise with at least 10 bullet points and it has to address the most important areas when it comes to building the product.
    """,
    agent=developer,
    expected_output="A detailed report with at least 10 bullet points addressing system design, technlogy and tools for building the protype."

)

crew = Crew(
    agents=[technologist,business_consultant,developer],
    tasks=[task1, task2, task3],
    verbose=True,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

result = crew.kickoff()

print("######################")
print(result)
