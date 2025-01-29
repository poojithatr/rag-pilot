import os
from typing import Dict, List
import boto3
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

# Configure AWS credentials
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# Initialize LLM options
mistral = OllamaLLM(model="mistral")
api_gemini = os.environ.get("GEMINI-API-KEY")
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)
openai_llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@tool
def get_compliance_status(admin_account_id: int) -> Dict:
    """
    Checks various compliance aspects for the AWS account, including:
    - Validating admin account access
    - Checking if MFA is enabled for all IAM users
    - Checking if CloudTrail logging is enabled
    - Listing AWS services in use and their configurations
    """
    try:
        # Create session with AWS credentials
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )

        # AWS Clients
        sts_client = session.client('sts')
        iam_client = session.client('iam')
        cloudtrail_client = session.client('cloudtrail')
        ec2_client = session.client('ec2')
        s3_client = session.client('s3')
        rds_client = session.client('rds')

        # Step 1: Validate the admin account
        caller_identity = sts_client.get_caller_identity()
        account_id = caller_identity['Account']
        if str(account_id).strip() != str(admin_account_id).strip():
            return {
                'status': 'error',
                'message': f"Session does not belong to the specified admin account. Connected to: {account_id}, Requested: {admin_account_id}"
            }

        # Step 2: Check MFA status for IAM users
        mfa_status = {}
        users = iam_client.list_users()['Users']
        for user in users:
            username = user['UserName']
            mfa_devices = iam_client.list_mfa_devices(UserName=username)['MFADevices']
            mfa_status[username] = len(mfa_devices) > 0

        # Step 3: Check CloudTrail logging status
        logging_enabled = False
        trails = cloudtrail_client.describe_trails()['trailList']
        for trail in trails:
            status = cloudtrail_client.get_trail_status(Name=trail['TrailARN'])
            if status.get('IsLogging'):
                logging_enabled = True
                break

        # Step 4: Collect configurations of active AWS services
        ec2_instances = [
            {"instance_id": instance['InstanceId'], "type": instance['InstanceType']}
            for reservation in ec2_client.describe_instances()['Reservations']
            for instance in reservation['Instances']
        ]

        s3_buckets = [{"bucket_name": bucket['Name']} for bucket in s3_client.list_buckets()['Buckets']]

        rds_databases = [
            {"db_instance_id": db['DBInstanceIdentifier'], "engine": db['Engine']}
            for db in rds_client.describe_db_instances()['DBInstances']
        ]

        # Step 5: Return consolidated data
        return {
            'status': 'success',
            'account_id': account_id,
            'mfa_status': mfa_status,
            'logging_enabled': logging_enabled,
            'ec2_instances': ec2_instances,
            's3_buckets': s3_buckets,
            'rds_databases': rds_databases
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }

# Define CrewAI Agents
data_collector = Agent(
    role="SOC2 Data Collector",
    goal="Interact with the AWS environment to gather comprehensive compliance data including MFA status, logging configurations, and service usage.",
    backstory="""You are responsible for collecting and analyzing AWS account compliance data, 
    including user security configurations, logging status, and service usage patterns. 
    You organize this data into a structured format for SOC2 compliance assessment.""",
    verbose=True,
    allow_delegation=True,
    tools=[get_compliance_status]
)

compliance_specialist = Agent(
    role="SOC2 Compliance Specialist",
    goal="Develop a comprehensive SOC2 compliance checklist on IAM roles, network security groups, ec2 insrances and logging and analyze collected data against the checklist.",
    backstory="You are an expert in SOC2 compliance and skilled at creating detailed checklists. You review parsed data and analyze it against compliance requirements.",
    verbose=True,
    allow_delegation=True
)

report_builder = Agent(
    role="SOC2 Compliance Report Builder",
    goal="Compile a detailed report based on the SOC2 compliance assessment data on IAM roles, network security groups, ec2 insrances and logging.",
    backstory="You create comprehensive reports highlighting compliance areas, gaps, and providing actionable recommendations.",
    verbose=True,
    allow_delegation=True
)

audit_qa_specialist = Agent(
    role="SOC2 Audit Q&A Specialist",
    goal="Answer questions about audit findings, reports, and compliance status.",
    backstory="You are an expert internal auditor with deep knowledge of SOC2 audit results and findings, capable of explaining technical details to various audiences.",
    verbose=True,
    allow_delegation=True
)

# Define Tasks
task1 = Task(
    description="""Connect to AWS and collect SOC2 compliance-related data on IAM roles, network security groups, ec2 insrances and logging.""",
    agent=data_collector,
    expected_output="A structured dataset of AWS configurations for compliance assessment."
)

task2 = Task(
    description="Analyze collected data against SOC2 compliance checklist and identify gaps data on IAM roles, network security groups, ec2 insrances and logging.",
    agent=compliance_specialist,
    expected_output="Detailed compliance analysis with identified gaps and assessment against SOC2 policy."
)

task3 = Task(
    description="Create detailed SOC2 compliance report with findings and recommendations data on IAM roles, network security groups, ec2 insrances and logging.",
    agent=report_builder,
    expected_output="Comprehensive compliance report with mapped gaps and actionable recommendations."
)

task4 = Task(
    description="Provide interactive Q&A support for audit findings and recommendations.",
    agent=audit_qa_specialist,
    expected_output="Detailed responses to compliance queries with references to specific findings."
)

if __name__ == "__main__":
    # Initialize CrewAI
    crew = Crew(
        agents=[data_collector, compliance_specialist, report_builder, audit_qa_specialist],
        tasks=[task1, task2, task3, task4],
        verbose=True,
        process=Process.sequential
    )

    try:
        # Execute the workflow
        result = crew.kickoff()
        print("\nCrewAI Execution Result:")
        print("######################")
        print(result)
        
    except Exception as e:
        print(f"\nError occurred:")
        print(f"Type: {type(e)}")
        print(f"Message: {str(e)}")
        import traceback
        print(f"\nTraceback:")
        print(traceback.format_exc()) 