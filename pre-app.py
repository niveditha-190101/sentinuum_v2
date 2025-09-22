## Llama index version of the code 
## Uses ReAct Agent with the functions as a tools 
## Updated solution 
## Should be the package for the streamlit app 
## rag part is in tools now 
## addedd context decision tool to decide whether conversation context is enough or if both conversation and transaction data are needed based on the issue and transcript.


from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from typing import List, Dict, Any

import os
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.prompts import PromptTemplate
from llama_index.core.readers.json import JSONReader
from llama_index.core.tools import FunctionTool

from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
import asyncio
import logging
import json
import os
from typing import Dict, Any
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from utils import AgentContext
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize LLM
llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Global context instance
agent_context = AgentContext()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

## creating and storing index from documents
    # load data
# collated_json = SimpleDirectoryReader(
#     input_files=["data/collated_v3.json"]
# ).load_data()
# transaction_doc = SimpleDirectoryReader(
#     input_files=["data/transaction.json"]
# ).load_data()

collated_json = SimpleDirectoryReader(
            input_files=["paysafe/conversation_data.json"]
        ).load_data()
transaction_doc = SimpleDirectoryReader(
            input_files=["paysafe/transaction data.json"]
        ).load_data()

# build index
collated_index = VectorStoreIndex.from_documents(collated_json)
transaction_index = VectorStoreIndex.from_documents(transaction_doc)
collated_engine = collated_index.as_query_engine(similarity_top_k=3)
transaction_engine = transaction_index.as_query_engine(similarity_top_k=3)

from llama_index.core.tools import QueryEngineTool



## function to load the collated json file and return a VectorStoreIndex of the json data for the RAG 
def load_and_index_json(json_path):
    reader = JSONReader(
        levels_back=0,
        collapse_length=None,
        ensure_ascii=False,
        is_jsonl=False,
        clean_json=True
    )
    
    documents = reader.load_data(input_file=json_path, extra_info={})
    index = VectorStoreIndex.from_documents(documents)
    return index 

## Funtcion to parse the rca response as seperate sections for the state 
def parse_rca_response_simple(response: str) -> dict:
    """
    Simple and effective RCA response parser.
    """
    analysis_results = {}
    current_key = None
    current_content = []
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('ROOT_CAUSE:'):
            # Store previous section if exists
            if current_key and current_content:
                analysis_results[current_key] = ' '.join(current_content)
            current_key = 'rootcause'
            current_content = [line.split('ROOT_CAUSE:', 1)[1].strip()]
            
        elif line.startswith('PATTERNS:'):
            if current_key and current_content:
                analysis_results[current_key] = ' '.join(current_content)
            current_key = 'pattern'
            current_content = [line.split('PATTERNS:', 1)[1].strip()]
            
        elif line.startswith('TIMELINE:'):
            if current_key and current_content:
                analysis_results[current_key] = ' '.join(current_content)
            current_key = 'timeline'
            current_content = [line.split('TIMELINE:', 1)[1].strip()]
            
        elif line.startswith('RATIONALE:'):
            if current_key and current_content:
                analysis_results[current_key] = ' '.join(current_content)
            current_key = 'rationale'
            current_content = [line.split('RATIONALE:', 1)[1].strip()]
            
        elif current_key:
            current_content.append(line)
    
    # Don't forget the last section
    if current_key and current_content:
        analysis_results[current_key] = ' '.join(current_content)
    
    return analysis_results
## Function to process transcript content directly instead of reading from a file
def process_transcript_content(transcript_content: str) -> str:
    """Process transcript content directly instead of reading from a file."""
    try:
        print(f"\n=== Process Transcript Debug ===")
        print(f"Processing transcript content")
            
        if not transcript_content:
            error_result = json.dumps({"error": "Transcript content is empty"})
            print(f"Output: {error_result}")
            return error_result
            
        result = json.dumps({
            "content": transcript_content,
            "source": "direct_input"
        }, ensure_ascii=False)
            
        print(f"Successfully processed transcript")
        print(f"Content length: {len(transcript_content)}")
        print(f"Output JSON format: {result[:200]}...")
        return result
            
    except Exception as e:
        error_result = json.dumps({"error": f"Error processing transcript: {str(e)}"})
        print(f"Error: {str(e)}")
        print(f"Output: {error_result}")
        return error_result
    finally:
        print("=== End Process Transcript Debug ===\n")




# MODIFIED TOOL FUNCTIONS TO USE CONTEXT 
## analyse _content to extract summary, name, issue, and status from transcript content
## rca to perform root cause analysis based on extracted name and issue
## check_context_status to check the current context status and determine next action
## decide_context_tool_fn to decide whether conversation context is enough or if both conversation and transaction data are needed based on the issue and transcript.
## run_conversation_tool to run the conversation context tool to get conversation context for a given issue for root cause analysis
## run_transaction_tool to run the transaction context tool to get transaction context for a given issue for root cause analysis



def decide_context_tool_fn() -> str:
    
    decision_prompt = f"""
    Based on the issue and transcript below, decide whether conversation context is enough
    or if both conversation and transaction data are needed.

    Issue: {agent_context.get('issue')}
    Transcript: {agent_context.get('summary')}

    Respond with: 'conversation_only' or 'both'
    """
    decision = llm.complete(decision_prompt)
    agent_context.update({"context_decision": decision.text.strip()})
    return decision
WORKFLOW_DIRECTORY = 'D:/vanguard/Sentiment Integrated/Conversations'

def run_conversation_tool() -> str:
    name = agent_context.get("name")
    issue = agent_context.get("issue")
    query = f"Customer: {name}. Issue: {issue}. Provide detailed conversation context."
    response = collated_engine.query(query)
    
    # You get the full node-level content (source, text, etc.)
    context_text = "\n\n".join([node.node.get_content() for node in response.source_nodes])
    
    agent_context.update({
        "conversation_context": context_text})
    
    return "Conversation context retrieved and stored."

def run_transaction_tool() -> str:
    name = agent_context.get("name")
    issue = agent_context.get("issue")
    query = f"Customer: {name}. Issue: {issue}. Provide detailed conversation context."
    response = transaction_engine.query(query)
    
    context_text = "\n\n".join([node.node.get_content() for node in response.source_nodes])
    
    # Add to agent context
    agent_context.update({
        "transaction_context": context_text})
    
    return "Transaction context retrieved and stored."
def analyze_content() -> Dict[str, Any]:
    """
    Analyze transcript content for summary, issue, and status using global context.
    """
    print("\n=== Starting Content Analysis ===")
    
    # Get transcript from context
    transcript = agent_context.get('transcript')
    
    if not transcript:
        error_msg = "No transcript content available in context"
        print(f"Error: {error_msg}")
        return {"error": error_msg}
    
    content_json = process_transcript_content(transcript)
    print(f"Content JSON: {content_json[:200]}...")

    if not content_json:
        print("Output: Empty input received")
        return {"error": "Empty input received"}

    try:
        content_data = json.loads(content_json)
        print("Successfully parsed JSON input")
    except json.JSONDecodeError as e:
        print(f"Initial JSON parse error: {str(e)}")
        content_data = {"content": content_json}
        print("Handled input as raw text")
    
    content = content_data.get("content", "")
    if not content:
        print("Output: No content found in input")
        return {"error": "No content found in input"}
    
    print(f"Content length for analysis: {len(content)}")
    print("Starting LLM analysis...")
    
    # Summary analysis
    summary_prompt_template = PromptTemplate(
        "Summarize the following customer service transcript in 2-3 sentences, focusing on the main issue and resolution:\n\n{content}\n\nSummary:"
    )
    formatted_prompt = summary_prompt_template.format(content=content)
    summary = llm.complete(formatted_prompt)
    print(summary.text)
    # Name extraction
    name_prompt = PromptTemplate(
        "Extract the customer's name from this transcript (not the agent). Return only the name:\n\n{content}\n\nCustomer Name:"
    )
    formatted_prompt = name_prompt.format(content=content)
    name = llm.complete(formatted_prompt)
    print("Name Identified")
    
    # Issue identification
    issue_prompt = PromptTemplate(
        "From the following transcript, identify the main issue in exactly 2-3 words:\n\n{content}\n\nMain Issue:"
    )
    formatted_prompt = issue_prompt.format(content=content)
    issue = llm.complete(formatted_prompt)
    print("Identified issue")
    
    # Status determination
    status_prompt = PromptTemplate(
        "From the following transcript, determine if the issue was 'resolved' or 'pending'. Return only one word:\n\n{content}\n\nStatus:"
    )
    formatted_prompt = status_prompt.format(content=content)
    status = llm.complete(formatted_prompt)
    print("Determined status")
    print(f"Status: {status.text.strip()}")
    
    # Update context with extracted data
    extracted_data = {
        "summary": str(summary.text).strip(),
        "name": str(name.text).strip(),
        "issue": str(issue.text).strip(),
        "status": str(status.text).strip(),
    }
    
    # Update global context
    agent_context.update(extracted_data)
    
    print(f"Analysis complete. Extracted: {list(extracted_data.keys())}")
    print("=== End Content Analysis ===\n")
    
    return {
        "message": "Content analysis completed successfully",
        "extracted_data": extracted_data,
        "context_status": "updated"
    }
from utils import apply_cleaning_transcripts,apply_matching_criteria,create_nodes_from_json,retrieve_context_from_nodes
def recommendation_retreiver()-> Dict[str, Any]:
    file_path = "data/collated_v3.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    cleaned_json = apply_cleaning_transcripts(json_data)
    filtered_json = apply_matching_criteria(cleaned_json, agent_context.get('issue'),"resolved")
    nodes = create_nodes_from_json(filtered_json)
    result = retrieve_context_from_nodes(nodes,agent_context.get("summary"))
    result_update = {"retrieved_information":result}
    agent_context.update(result_update)
    return {
        "message": "Recommendation context retrieved and stored",}

# def rca() -> Dict[str, Any]:
#     """
#     Performs RAG-based root cause analysis using the knowledge base and context data.
#     """
#     print("\n=== Starting Root Cause Analysis ===")
    
#     # Get required data from context
#     name = agent_context.get('name')
#     issue = agent_context.get('issue')
    
#     if not name or not issue:
#         missing = []
#         if not name: missing.append("name")
#         if not issue: missing.append("issue")
#         error_msg = f"Missing required data for RCA: {missing}"
#         print(f"Error: {error_msg}")
#         return {"error": error_msg}
    
#     print(f"Performing RCA for: {name} - {issue}")
    
#     # Load and prepare data
#     file_path = "D:/vanguard/Sentiment Integrated/collated_v2.json"
#     transaction_data = 'D:/vanguard/Sentiment Integrated/transaction.json'
    
#     try:
#         # Load and index data
#         loader = load_and_index_json(file_path)
#         loader_transaction = load_and_index_json(transaction_data)
        
#         # Set up retrievers
#         retriever = loader.as_retriever(similarity_top_k=2)
#         transaction_retriever = loader_transaction.as_retriever(similarity_top_k=2)
        
#         # Perform retrieval
#         query = f"{name} {issue}"
#         context_nodes = retriever.retrieve(query)
#         transaction_nodes = transaction_retriever.retrieve(query)
        
#         # Extract and join text content from the retrieved nodes
#         context = "\n".join([node.node.text for node in context_nodes])
#         transaction_context = "\n".join([node.node.text for node in transaction_nodes])

#         template = """
#         Knowledge Base Context: {context}
#         Transaction Context: {transaction_context}
        
#         Analysis Instructions
#         You are tasked with performing a comprehensive root cause analysis by connecting information between the knowledge base context and transaction data. Your goal is to identify patterns, relationships, and causal factors that explain the situation.
        
#         Required Analysis Tasks for:
#         Name: {name}
#         Issue: {issue}
        
#         1. Timeline Integration Analysis
#         Extract all timestamps from both the knowledge base context and transaction data
#         Create a unified chronological timeline incorporating both data sets
#         Identify temporal correlations between customer interactions/issues and financial transactions
#         Examine transaction patterns (withdrawals, deposits, liquidations) in relation to key events
#         Complete the timeline in 6 exact sentences include important events
        
#         2. Root Cause Identification
#         Explain everything in 5 W's: why is the customer dissatisfied, what is the issue, what steps are taken, what is the result
        
#         3. Pattern Recognition
#         Identify recurring patterns or trends across the data
#         Note specific triggers that appear to initiate transaction behaviors
#         Identify common failure points where customer issues and transaction problems intersect
#         Detect unusual transaction patterns and their relation to customer interactions
        
#         REQUIRED OUTPUT FORMAT
#         Your analysis must be provided in exactly the following format with these precise section headers:
        
#         ROOT_CAUSE: Explain everything in 5 W's: why is the customer dissatisfied, when did this happen, what is the issue, what steps are taken, what is the result?
        
#         PATTERNS: Detail the recurring patterns discovered across both datasets. Include specific triggers, common failure points, and any unusual transaction behaviors (continuous withdrawals/deposits, liquidation patterns, etc.) that correlate with customer interactions. Highlight significant transaction anomalies and their connection to customer issues. Explain all these in 5 sentences short and concise.
        
#         TIMELINE: Present a chronological organization of all relevant events from both datasets. Include specific dates from both contexts, showing how they correlate. Highlight important milestones, decision points, initial contacts, escalations, follow-up actions, and resolution attempts. Show clear connections between dates in the knowledge base and transaction activities. 6 sentences from this timeline having the important information. Stick to this format example September 10th, 2023.
        
#         RATIONALE: Explain your reasoning process and why you believe these connections exist between the knowledge base context and transaction data. Include your analysis of how customer behavior correlates with transaction activities and what process failures might be occurring. Provide evidence-based justification for your conclusions and show me the numbers from the transaction data that is related to the analysis. Also provide few actionable recommendations.
        
#         Important Notes
#         Each section must start exactly with the section header (ROOT_CAUSE:, PATTERNS:, etc.) followed by your analysis
#         Be comprehensive but concise in your analysis
#         Ensure you're making explicit connections between the knowledge base context and transaction context
#         Focus on specific evidence and avoid general statements
#         Include all relevant dates and look for temporal correlations between contexts
#         """
        
#         prompt = PromptTemplate(template)
        
#         # Format the template with your variables
#         formatted_prompt = prompt.format(
#             context=context,
#             transaction_context=transaction_context,
#             name=name,
#             issue=issue
#         )
        
#         # Use the formatted prompt with your LLM
#         response = llm.complete(formatted_prompt)
#         rca_result = str(response.text)
#         analysis_results =  parse_rca_response_simple(rca_result)
        
#         print(rca_result)
        
#         # Store RCA result in context
#         agent_context.update(analysis_results)
        
#         print("RCA completed successfully")
#         print("=== End Root Cause Analysis ===\n")
        
#         return {
#             "message": "Root cause analysis completed successfully",
#             "rca_result": rca_result[:500] + "..." if len(rca_result) > 500 else rca_result,
#             "context_status": "updated"
#         }
    
#     except Exception as e:
#         error_msg = f"Error during RCA: {str(e)}"
#         print(f"Error: {error_msg}")
#         return {"error": error_msg}



def perform_root_cause_analysis() -> dict:
    prompt = f"""
    Perform root cause analysis for the issue.

    Name: {agent_context.get('name')}
    Issue: {agent_context.get('issue')}
    Conversation Context: {agent_context.get('conversation_context') if agent_context.get('context_decision') == 'conversation_only' else ""}
    {"Transaction Context: " + str(agent_context.get('transaction_context')) if agent_context.get('context_decision') == 'both' else ""}
    1. Timeline Integration Analysis
        Extract all timestamps from both the knowledge base context and transaction data
        Create a unified chronological timeline incorporating both data sets
        Identify temporal correlations between customer interactions/issues and financial transactions
        Examine transaction patterns (withdrawals, deposits, liquidations) in relation to key events
        Complete the timeline in 6 exact sentences include important events
        
        2. Root Cause Identification
        Explain everything in 5 W's: why is the customer dissatisfied, what is the issue, what steps are taken, what is the result
        
        3. Pattern Recognition
        Identify recurring patterns or trends across the data
        Note specific triggers that appear to initiate transaction behaviors
        Identify common failure points where customer issues and transaction problems intersect
        Detect unusual transaction patterns and their relation to customer interactions
    
    REQUIRED OUTPUT FORMAT
        Your analysis must be provided in exactly the following format with these precise section headers:
        
        ROOT_CAUSE: Explain everything in 5 W's: why is the customer dissatisfied, when did this happen, what is the issue, what steps are taken, what is the result?
        
        PATTERNS: Detail the recurring patterns discovered across both datasets. Include specific triggers, common failure points, and any unusual transaction behaviors (continuous withdrawals/deposits, liquidation patterns, etc.) that correlate with customer interactions. Highlight significant transaction anomalies and their connection to customer issues. Explain all these in 5 sentences short and concise.
        
        TIMELINE: Present a chronological organization of all relevant events from both datasets. Include specific dates from both contexts, showing how they correlate. Highlight important milestones, decision points, initial contacts, escalations, follow-up actions, and resolution attempts. Show clear connections between dates in the knowledge base and transaction activities. 6 sentences from this timeline having the important information. Stick to this format example September 10th, 2023.
        
        RATIONALE: Explain your reasoning process and why you believe these connections exist between the knowledge base context and transaction data. Include your analysis of how customer behavior correlates with transaction activities and what process failures might be occurring. Provide evidence-based justification for your conclusions and show me the numbers from the transaction data that is related to the analysis. Also provide few actionable recommendations.
        
        Important Notes
        Each section must start exactly with the section header (ROOT_CAUSE:, PATTERNS:, etc.) followed by your analysis
        Be comprehensive but concise in your analysis
        Ensure you're making explicit connections between the knowledge base context and transaction context
        Focus on specific evidence and avoid general statements
        Include all relevant dates and look for temporal correlations between contexts
        
    
    """

    
   
    response = llm.complete(prompt)
    print(response.text)
    data = {"rootcause": str(response.text)}
    agent_context.update(data)
    return {
            "message": "Root cause analysis completed successfully",

            "context_status": "updated"
        } 

def recommendation () -> Dict[str, Any]:
    # query str should be updated from context 
    # context_str should be updated from context 
    prompt = (f"""
    You are a customer service assistant. Your task is to generate thoughtful and actionable recommendations to help resolve or prevent the issue described below. 
        Use only the context from similar past resolved cases.
              
    Issue: {agent_context.get('summary')}

    Knowledge Base Context (from resolved cases only): {agent_context.get('retrieved_information')}    


    Instructions:
            Do NOT treat any information as directly related to the current user unless clearly applicable to the issue.
            Avoid suggesting actions based on unresolved or irrelevant conversations.
            Recommend specific actions based on how similar issues were successfully handled.
            Tailor the suggestions closely to the patterns, actions, and outcomes observed in these resolved cases.
            Be clear, concise, and customer-focused. 
            Avoid vague or generic advice.

    Provide your response in the following format:

    RECOMMENDATIONS:
            1. ...
            2. ...
            3. ...
            ...
    """)

    # prompt = (f"""
    # You are a customer service assistant. Your task is to generate thoughtful, actionable, and prioritized recommendations to help resolve or prevent the issue described below.
    #     Use only the context from similar past resolved cases.

    #     Issue:
    #         {agent_context.get('summary')}

    #     Knowledge Base Context (from resolved cases only):
    #         {agent_context.get('retrieved_information')}

    #     Instructions:
    #     - Do NOT treat any information as directly related to the current user unless clearly applicable to the issue.
    #     - Avoid suggestions based on unresolved, irrelevant, or inconclusive cases.
    #     - Recommend specific actions that were shown to be successful in similar resolved situations.
    #     - Tailor suggestions based on real patterns of resolution — what worked, how it worked, and why.
    #     - Be clear, concise, and practical. Avoid vague or generic advice.

    #     Scoring & Ranking Logic:
    #     Evaluate and rank the recommendations using the following criteria assigning an **overall score of 100**:
    #         • Effectiveness(40%) : How well did this action resolve similar issues?
    #         • Relevance(25%) : How closely does this action apply to the current issue?
    #         • Impact(15%) : Did this action lead to clear improvements or customer satisfaction?
    #         • Feasibility(10%) : Can this action be implemented easily and quickly?
    #         • Frequency(10%) : Has this action been successfully used across multiple cases?

    #     Prioritize the recommendations based on their combined strength across these factors.

    #     Format your response like this:

    #     RECOMMENDATIONS (ranked by their scores):
    #     1. ...
    #     2. ...
    #     3. ...
    #     ...
    #     """)


    # prompt_template = PromptTemplate(prompt)
    # you can create text prompt (for completion API)
    # llm = prompt_template.format_messages(context_str=retrieved_information, query_str=customer_concern_summary)
    # print("Prompt provided to LLM - augmented with relevant context", llm_prompt[0],sep="\n\n")
    response=llm.complete(prompt)
    print("LLM Response ",response.text,sep="\n\n")
    ## add recomnedation context and then add this set the variable there 
    return response.text

def check_context_status() -> Dict[str, Any]:
    """
    Check the current context status and determine next action.
    """
    print("\n=== Checking Context Status ===")
    
    try:
        # Define required fields for basic data extraction
        required_fields = ['name', 'issue', 'status']  # Add other required fields as needed
        missing_fields = []
        
        # Check for missing fields manually
        for field in required_fields:
            try:
                value = agent_context.get(field)
                if not value or (isinstance(value, str) and value.strip() == ''):
                    missing_fields.append(field)
            except Exception as e:
                print(f"Error getting field '{field}': {e}")
                missing_fields.append(field)
        
        if missing_fields:
            print(f"Missing fields: {missing_fields}")
            return {
                'next_action': 'extract_data',
                'status': 'incomplete',
                'missing_fields': missing_fields,
                'message': f"Need to extract: {', '.join(missing_fields)}"
            }
        
        # Check if RCA is needed
        try:
            rootcause = agent_context.get('rootcause')
        except Exception:
            rootcause = None
            
        if not rootcause:
            try:
                status = agent_context.get('status')
                if not status:
                    status = ''
                status = str(status).lower()
                
                if any(keyword in status for keyword in ['pending', 'open', 'unresolved', 'escalated']):
                    print("RCA needed due to unresolved status")
                    return {
                        'next_action': 'perform_rca',
                        'status': 'ready_for_rca',
                        'message': 'All data extracted. Ready for root cause analysis.'
                    }
                elif any(keyword in status for keyword in ['resolved', 'closed', 'completed']):
                    print("Issue resolved - RCA optional")
                    return {
                        'next_action': 'perform_rca',  # Still perform RCA for learning
                        'status': 'resolved_but_rca_recommended',
                        'message': 'Issue resolved but RCA recommended for process improvement.'
                    }
                else:
                    # If status is unclear, default to RCA
                    print("Status unclear - defaulting to RCA")
                    return {
                        'next_action': 'perform_rca',
                        'status': 'status_unclear',
                        'message': 'Status unclear. Performing RCA for comprehensive analysis.'
                    }
            except Exception as e:
                print(f"Error checking status: {e}")
                return {
                    'next_action': 'perform_rca',
                    'status': 'status_check_failed',
                    'message': 'Could not determine status. Performing RCA for comprehensive analysis.'
                }
        
        # Everything is complete
        print("All analysis complete")
        
        # Safely get summary data
        summary_data = {}
        for field in ['name', 'issue', 'status']:
            try:
                summary_data[field] = agent_context.get(field)
            except Exception:
                summary_data[field] = None
        
        try:
            rca_performed = bool(agent_context.get('rootcause'))
        except Exception:
            rca_performed = False
            
        return {
            'next_action': 'complete',
            'status': 'complete',
            'message': 'All analysis completed successfully.',
            'summary': {
                **summary_data,
                'rca_performed': rca_performed
            }
        }
        
    except Exception as e:
        error_msg = f"Error checking context status: {e}"
        print(error_msg)  # Using print instead of logging for consistency
        return {
            'next_action': 'extract_data',
            'status': 'error',
            'error': error_msg,
            'message': 'Error occurred. Defaulting to data extraction.'
        }
    finally:
        print("=== End Context Status Check ===\n")
        
        
        

### Defining the tools for the ReAct agent
tools = [
    FunctionTool.from_defaults(
        fn=analyze_content, 
        name="analyze_content", 
        description="Analyze transcript content for summary, name, issue, and status. Call this first to extract data from transcript."
    ),
    FunctionTool.from_defaults(
        fn=perform_root_cause_analysis, 
        name="root_cause_analysis", 
        description="Perform comprehensive root cause analysis based on extracted name and issue. Call this after analyze_content."
    ),
    FunctionTool.from_defaults(
        fn=check_context_status, 
        name="check_context_status", 
        description="Check the current context status and get next recommended action. Use this to determine workflow progress."
    ),
    FunctionTool.from_defaults(
        fn=run_conversation_tool, 
        name="run_conversation_tool", 
        description="Run the conversation context tool to get conversation context for a given issue for root cause analysis."
    ),
    FunctionTool.from_defaults(
        fn=run_transaction_tool, 
        name="run_transaction_tool", 
        description="Run the transaction context tool to get transaction context for a given issue for root cause analysis"
    ),
    FunctionTool.from_defaults(
        fn=decide_context_tool_fn,
        name="decide_context_tool",
        description="Decide whether conversation context is enough or if both conversation and transaction data are needed based on the issue and transcript."
    ),
    FunctionTool.from_defaults(
        fn=recommendation_retreiver,
        name="recommendation_retreiver",
        description ="Retreives the important context needed for the recommendation tool based on the issue and context. Use this to provide specific, evidence-based suggestions for resolution or prevention."
    ),
    
    FunctionTool.from_defaults(
        fn=recommendation,
        name= "recommendation",
        description="Generate actionable recommendations based on the issue and context. Use this to provide specific, evidence-based suggestions for resolution or prevention."
    )
    
]
## Defining the agent context with autonomous capabilities
agent_context = """
You are an autonomous Customer Service Analysis Agent.

Your job is to analyze transcripts and identify root causes behind customer issues.

Follow this strict tool usage order:

1. ALWAYS call 'analyze_content' FIRST to extract name, issue, and status.
2. Then call 'check_context_status'. If the issue is resolved, perfrom RCA for learning purpose.
3. If RCA is needed, call 'decide_context_tool' to determine what data is required.
4. Based on that decision:
   - Call 'run_conversation_tool' if only conversation context is needed.
   - Call BOTH 'run_conversation_tool' and 'run_transaction_tool' if full context is needed.
5. ONLY AFTER the proper context is available, call 'perform_root_cause_analysis'.
6. After RCA, call 'recomendation_retreiver' to gather relevant context for recommendations.
7. Finally, call 'recomendation' to generate actionable recommendations based on the issue and context.

NEVER call 'perform_root_cause_analysis' without first ensuring the required context is loaded.

Be auto

You are fully autonomous. Use tools as needed, rerun steps when necessary, and ensure your insights drive measurable improvements in customer satisfaction and operational efficiency.
- If the root cause is unclear after 2 context retrievals, flag the case as "NEEDS MANUAL REVIEW".
- Annotate any assumptions made due to missing context.
- Log intermediate tool decisions and final context used in each analysis.
"""

agent = ReActAgent(
    llm=OpenAI(
        model="gpt-4o", 
        temperature=0.1, 
        max_tokens=4000,  
        timeout=60.0      
    ), 
    tools=tools,
    memory=ChatMemoryBuffer.from_defaults(
        token_limit=4000,  
    ),
    max_iterations=25,  
    context=agent_context,
    verbose=True
)

# from streamlit_base import load_transcript_files,get_folder_structure
async def main():
    print("Starting ReAct Agent Workflow with Context Management")
    print("=" * 60)
    global agent_context
    agent_context = AgentContext()  # Reset context for fresh run
    # Initialize context with transcript
    transcript_content = """
Good morning, Mr. Wilson. This is David Butler from customer service. I'm calling about your savings account case 298745613092 regarding duplicate charges on your account. How can I help you today?
David, I'm incredibly frustrated. My savings account has been charged twice for the same transactions multiple times this month, and it's creating a serious overdraft situation that shouldn't exist.
I apologize for the inconvenience, Mr. Wilson. Let me pull up your account activity right away. I can see your savings account ending in 3092, and I'll review the recent transactions to identify the duplicate charges.
There are at least four instances where I was charged twice for the same purchase. A $85 grocery store charge, a $150 gas station fill-up, and two different restaurant bills - all duplicated within minutes of each other.
I can see exactly what you're referring to, Mr. Wilson. There are indeed duplicate charges showing on your account - the same merchants, same amounts, processed within 2-3 minutes of each original transaction.
This has caused my savings account to go into overdraft because the duplicate charges pushed me over my available balance. Now I'm being charged overdraft fees on top of the duplicate charges!
That's completely unfair, Mr. Wilson. I can see three overdraft fees totaling $105 that were triggered directly by these duplicate charges. This is clearly a processing error on our end, not your spending.
Exactly! I carefully manage my finances and never overdraw my accounts. These duplicate charges have cost me nearly $400 in fake transactions plus your overdraft penalties.
I understand your frustration completely. These appear to be system processing errors where transactions were submitted twice by our card network. I'll need to dispute each duplicate charge individually through our systems.
How long is that going to take? I have automatic payments scheduled for next week, and if this isn't fixed, I'll get hit with even more overdraft fees for legitimate transactions.
The dispute process for duplicate charges typically takes 5-7 business days per transaction. Since you have four duplicates, this could take up to two weeks to fully resolve all the reversals.
Two weeks? That's completely unacceptable! These are obvious duplicate charges - same merchant, same amount, same time. Why can't you just reverse them immediately?
I wish I could reverse them instantly, Mr. Wilson, but our system requires each duplicate to go through the formal dispute process for regulatory compliance, even when the error is clearly on our side.
So I'm supposed to suffer for two weeks because of your bank's processing errors? What about my upcoming automatic payments that will bounce because of this mess?
That's a valid concern, Mr. Wilson. I can try to get a temporary credit approved to cover the duplicate amounts while the disputes are processing, but that also requires manager approval and could take 24-48 hours.
Twenty-four to forty-eight hours for a temporary credit on your own mistake? This is the worst customer service I've ever experienced. I'm considering closing all my accounts.
I don't want to lose you as a customer, Mr. Wilson. Let me escalate this to my supervisor immediately and see if we can expedite both the temporary credits and the permanent dispute resolutions.
I need more than promises, David. I need concrete action and a timeline that actually makes sense for what should be a simple fix to an obvious bank error.
You're absolutely right, Mr. Wilson. Let me schedule a follow-up call for tomorrow afternoon so I can provide you with definitive updates on both the temporary credits and the dispute timeline after speaking with my supervisor.
Fine, but if this isn't significantly improved by tomorrow, I'm filing complaints with banking regulators and moving my money to a bank that can handle basic transaction processing correctly.
I understand your frustration entirely, Mr. Wilson. I'll call you tomorrow at 2 PM with concrete resolutions and ensure we prevent this type of processing error from affecting your account in the future.
"""
    try:
        # Initialize context with transcript
        agent_context.initialize(transcript_content)
        
        print(f"Context initialized with transcript length: {len(transcript_content)}")
        print(f"Initial context state: {list(agent_context.get_all().keys())}")
        
        # Simulate some context updates
        agent_context.set("name", "John Doe")
        agent_context.set("issue", "Account login problem")
        agent_context.set("summary", "Customer unable to login to account")
        agent_context.set("status", "In Progress")
        
        # Check if context is complete
        print(f"\nContext complete: {agent_context.is_complete()}")
        print(f"Missing fields: {agent_context.get_missing_fields()}")
        agent_context.initialize(transcript_content)
        
        print(f"Context initialized with transcript length: {len(transcript_content)}")
        print(f"Initial context state: {list(agent_context.get_all().keys())}")
      
        handler = agent.chat(
            message="""
            I have a transcript of a customer service interaction that has been loaded into the context.
            Please follow this workflow:
            Please execute each step methodically and report on your progress at each stage.
            Use the tools provided and follow the workflow exactly as described.
            """
        )
        
        response = handler
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print("=" * 60)
        print(f"Agent Response: {str(response)}")
        
        print("\nFinal Context State:")
        final_context = agent_context.get_all()
        for key, value in final_context.items():
            if value is not None:
                display_value = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                print(f"  {key}: {display_value}")
            else:
                print(f"  {key}: None")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())