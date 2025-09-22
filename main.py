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

# ## creating and storing index from documents
#     # load data
# collated_json = SimpleDirectoryReader(
#     input_files=["data/collated_v2_copy.json"]
# ).load_data()
# transaction_doc = SimpleDirectoryReader(
#     input_files=["data/transaction.json"]
# ).load_data()

## creating and storing index from documents
    # load data
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
    # file_path = "data/collated_v2_copy.json"
    file_path = r"C:\Users\niveditha.n.lv\Documents\summarizer\paysafe\conversation_data.json"
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
#     transcript_content = """Welcome to Guardian Investment Services, this is Rachel speaking. How may I help you today?
# Hi Rachel, this is Daniel Patel. I'm calling because I'm having serious issues with delayed transactions in my account. This has been going on for days now.
# I apologize for any inconvenience, Mr. Patel. Let me verify your account information first. Could you please provide your account number and the last four digits of your SSN?
# Yes, account number is 775-90142, last four are 6589.
# Thank you for verifying that. Could you tell me more about the delays you're experiencing?
# Well, I initiated three wire transfers last week - two outgoing and one incoming. The money left the sending accounts days ago, but they're still not showing up. And now I'm getting hit with overdraft fees because of this!
# I understand your frustration. Let me check those transfers... I see them here. They were initiated on February 13th. There seems to be a processing delay due to... hmm.
# Due to what? I need that money available now. This is affecting my business operations.
# I see that there was a flag raised by our compliance department. Sometimes this happens with multiple simultaneous transfers. Unfortunately, they need to review each transaction.
# But why wasn't I notified about this? I've been calling daily and nobody mentioned anything about compliance reviews.
# I sincerely apologize for the lack of communication. You should have been informed. Let me check the status of the review... 
# This is ridiculous. I've been banking with you for five years, and these are normal transactions for my business account.
# I completely understand. The outgoing transfers are for $45,000 and $32,000, and the incoming one is for $68,000, correct?
# Yes, that's right. And every day they're delayed costs me money and credibility with my vendors.
# I'm escalating this to our priority review team right now. However, even with expedited processing, it might take another 24-48 hours to complete the review.
# That's not acceptable. I can't keep doing business like this. I want to close my account as soon as these transfers clear.
# I understand your decision, Mr. Patel. While we value your business greatly, I can help you initiate the account closure process. Would you like me to send you the necessary forms?
# Yes, please. And I want written confirmation about these delays and why they happened. I've lost thousands in overdraft fees and delayed payments.
# Of course. I'll document everything in detail, including the timeline of the transfers and the compliance review. I'll also request a review of any fees incurred due to these delays.
# What about the immediate situation? I need these transfers processed.
# I'm adding an urgent flag to your case right now. You should receive a call from our priority review team within the next hour. They can better explain the specific requirements to clear these transfers.
# Fine. Please make sure they call. And send me those closure forms right away.
# I'll email everything to you within the next 15 minutes. Is your email still dpatel@email.com?
# Yes, that's correct.
# I'll send all the documentation and forms right away. Again, I sincerely apologize for this situation. Is there anything else you need assistance with?
# No, just make sure the review team calls me within the hour.
# They will definitely call you. Thank you for your patience, Mr. Patel. I understand how frustrating this has been.
# Thank you. Goodbye."""
    transcript_content = """
Conversation 1 Date – July 3, 2025
Merchant:
Hi, I received a chargeback notification today for ₹18,500 that says it's unpaid. But I paid it on July 3rd morning through NEFT. I even have the UTR: AXIS98382122.
Agent:
Thank you, Mr. Sinha. Let me pull up your account and verify... Yes, I see the chargeback listed under dispute ID CB18507, but it’s still showing as “pending”. If you’ve made the payment, it may not have been reconciled yet in the backend.
Merchant:
That makes no sense. It's been nearly 24 hours. The payment is settled from my bank.
Agent:
Understood. I’ll raise a service ticket now and attach the UTR to the dispute. You’ll get a response within 2-3 working days. I’ve also marked this as "Urgent".
Merchant:
You better fix this. I got an SMS saying my merchant account will be suspended if I don’t pay. I have paid.

Conversation 2 Date – July 6, 2025
Merchant:
Hello, this is Arvind Sinha. I’m following up on a chargeback issue from three days ago. Ticket number is SR45298. I paid the chargeback and submitted proof.
Agent:
Let me check that… I’m sorry sir, I don’t see that ticket in our current queue. Could you explain the issue again?
Merchant:
What do you mean? I explained all this to a guy named Prakash. I even gave the UTR. Are your support systems not connected?
Agent:
I apologize. Sometimes older tickets are routed to another internal desk, and we may not have visibility. Can you share the UTR again?
Merchant:
AXIS98382122! This is the second time I’ve repeated this. I don’t want another apology—I want this fixed.
Agent:
Completely understand, sir. I’ll raise a new ticket (SR46003) and escalate it again to Risk & Reconciliation.

Conversation 3 Date– July 8, 2025
Merchant:
I’ve had enough of this. I’ve submitted payment proof twice. Today I got another warning from the Payment Bank that my account will be locked. This is ridiculous.
Agent:
I’m really sorry, Mr. Sinha. Let me check... Okay, I see both SR45298 and SR46003 were opened. According to the backend note, your payment was received but tagged under a different chargeback ID. That’s why it’s still marked “unpaid”.
Merchant:
So your backend misfiled the reference, and I’m the one getting flagged?
Agent:
Unfortunately, yes sir. I’ve just now requested the Settlement Operations team to re-map your payment to the correct dispute ID. I’ll follow up personally and confirm once it’s done.
Merchant:
You need to make sure this gets fixed today. I won’t let my account get blocked because of your internal error.

Conversation 4 Date – July 9, 2025
Merchant:
I just got a call saying my merchant account will be suspended in 24 hours. What the hell is going on?
Agent:
Sir, I’m really sorry. I see a pending alert that was automatically triggered due to the open chargeback flag. It was not manually reviewed after your previous escalation.
Merchant:
I’ve spoken to THREE agents already. Why does no one have any context? I’ve given UTR, I’ve raised tickets, I’ve received empty promises.
Agent:
Completely valid complaint, sir. I’m freezing the suspension temporarily and escalating this to our internal risk head for manual override. You won’t face further action while this is pending.
Merchant:
This is the fourth time I’ve explained the same thing. Do you guys not use a CRM? How is every new agent clueless?
Agent:
I understand your frustration. I’ll submit a complaint regarding the repeated re-briefing as well. Please expect a senior-level callback.

Conversation 5 Date– July 10, 2025
Merchant:
Now there's a NEW problem. A customer payment of ₹72,000 made on July 9 via UPI isn’t reflecting in my settlement report.
Agent:
That’s concerning. Let me check your settlement records… I don’t see that order in today’s payout batch. Could you share the order ID?
Merchant:
Order ID is MTR-99012. My customer’s account was debited at 10:17 AM. Where is the money?
Agent:
It could be a reconciliation issue, especially if the UPI ID wasn’t mapped correctly. I’ll raise a new ticket for this.
Merchant:
So now in addition to the chargeback fiasco, you’ve lost a customer’s ₹72,000 payment? I can’t afford this kind of incompetence.
Agent:
Completely understandable, sir. I’ve raised ticket #SR47300 for this payout issue and marked it as “Critical”. We’ll prioritize resolution.

Conversation 6 Date – July 11, 2025
Subject: Escalation call from support manager
Escalation Manager:
Mr. Sinha, this is Neha from Merchant Support Escalations. I’ve reviewed all your previous tickets and calls personally.
Merchant:
I hope you finally have a full picture. I’ve spoken to five different people, repeated the same story every time. Nothing’s fixed.
Escalation Manager:
You’re absolutely right, and we take full responsibility. To summarize:
Your chargeback of ₹18,500 was settled on July 3, but it was incorrectly mapped and flagged as unpaid. That flag has now been removed and reversed in the system.
Your ₹72,000 customer payment on July 9 was delayed due to an error in batch reconciliation. That payment has now been credited as of 3:30 PM today.
Our agents should have had access to your ticket history. That failure is unacceptable. We are reviewing our support workflow immediately.
Merchant:
You’ve put my business at risk. You’ve wasted my time and pushed me to the edge. One more issue like this and I go to the Banking Ombudsman.
Escalation Manager:
I completely understand. As a gesture of apology we are crediting ₹2,000 in your merchant wallet and MDR fees will be waived for the next billing cycle. You’ll receive an official apology email and case summary within 24 hours."""


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

    
# def display_beautified_transcript(tagged_output):
#     """Display tagged transcript in a beautifully formatted way"""
#     st.subheader("Conversation Analysis")
        
#     try:
#         # Parse the tagged output if it's a string
#         if isinstance(tagged_output, str):
#             try:
#                 data = json.loads(tagged_output)
#             except json.JSONDecodeError:
#                 st.error("Unable to parse the tagged output as JSON")
#                 st.text_area("Raw Tagged Output", value=tagged_output, height=500)
#                 return
#         else:
#             data = tagged_output
            
#         # Display Metadata in a nice card
#         if 'conversation_metadata' or 'analytics' in data:
#             metadata = data['conversation_metadata']
#             # analytics = data['analytics']['overall_sentiment']
#             analytics = data['analytics']
#             sentiment = analytics.get('overall_sentiment', 'N/A')
#             sentiment_color = get_sentiment_color(sentiment)
#             st.markdown("### Metadata")
#             # st.markdown(f"**Overall Sentiment:** {analytics.get('overall_sentiment', 'N/A')}")
#             st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color};'>{sentiment}</span>", 
#                         unsafe_allow_html=True)
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown(f"**Customer:** {metadata.get('customer_name', 'N/A')}")
#                 st.markdown(f"**Account Type:** {metadata.get('account_type', 'N/A')}")
#                 st.markdown(f"**Account Number:** {metadata.get('account_number', 'N/A')}")
#             with col2:
#                 st.markdown(f"**Call Type:** {metadata.get('call_type', 'N/A')}")
#                 st.markdown(f"**Agent Name:** {metadata.get('agent_name', 'N/A')}")
            
#         # Display Summary
#         if 'conversation_details' in data and 'summary' in data['conversation_details']:
#             summary = data['conversation_details']['summary']
#             st.markdown("### High-level Summary")
#             st.markdown(f"**Overview:** {summary.get('overview', 'N/A')}")
                
#             # Summary details in columns
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown(f"**Primary Issue:** {summary.get('primary_issue', 'N/A')}")
#             with col2:
#                 st.markdown(f"**Resolution Status:** {summary.get('resolution_status', 'N/A')}")
#                 st.markdown(f"**Call Outcome:** {summary.get('call_outcome', 'N/A')}")
            
#         # Display Transcript Segments as a timeline
#         if 'conversation_details' in data and 'transcript_segments' in data['conversation_details']:
#             segments = data['conversation_details']['transcript_segments']
#             with st.expander("### Timeline", expanded=False):
                
#                 for idx, segment in enumerate(segments):
#                     # Create a container for each message
#                     with st.container():
#                         # Determine if it's customer or agent for styling
#                         is_customer = segment.get('speaker', '').lower() == 'customer'
#                         speaker_color = "rgba(0, 120, 212, 0.1)" if is_customer else "rgba(100, 100, 100, 0.1)"
#                         speaker_name = segment.get('speaker', 'Unknown').capitalize()
                            
#                         # Format timestamp if available
#                         timestamp_str = ""
#                         if 'timestamp' in segment:
#                             try:
#                                 # Convert ISO timestamp to readable format
#                                 dt = datetime.fromisoformat(segment['timestamp'].replace('Z', '+00:00'))
#                                 timestamp_str = dt.strftime("%H:%M:%S")
#                             except:
#                                 timestamp_str = segment['timestamp']

#                         st.markdown(
#                             f"""
#                                     <div style="
#                                     background-color: {speaker_color}; 
#                                     padding: 10px 15px; 
#                                     border-radius: 10px; 
#                                     margin: 5px 0;
#                                     max-width: 80%;
#                                     {'' if is_customer else 'margin-left: 20%;'}
#                                     ">
#                                     <strong>{speaker_name}</strong> <span style="color: gray; font-size: 0.8em;">({timestamp_str})</span>
#                                     <p style="margin: 5px 0 0 0;">{segment.get('text', '')}</p>
#                                     {f'<div style="font-size: 0.8em; margin-top: 5px;"><em>Sentiment: <span style="color: {"red" if segment.get("sentiment") == "negative" else "green" if segment.get("sentiment") == "positive" else "gray"}">{segment.get("sentiment", "neutral")}</span></em></div>' if 'sentiment' in segment else ''}
#                                     {f'<div style="font-size: 0.8em; margin-top: 2px;"><em>Key phrases: {", ".join([f"<strong>{phrase}</strong>" for phrase in segment.get("key_phrases", [])])}</em></div>' if segment.get('key_phrases') else ''}
#                                     </div>
#                                     """, 
#                                     unsafe_allow_html=True
#                                     )
                
#     except Exception as e:
#             st.error(f"Error displaying beautified transcript: {str(e)}")
#             # Fallback to raw display
#             st.text_area("Raw Tagged Output", value=str(tagged_output), height=500)   