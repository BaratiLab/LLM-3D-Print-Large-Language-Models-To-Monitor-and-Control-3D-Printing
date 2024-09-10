from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Image, display
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from langchain.agents import AgentExecutor
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser,OutputFunctionsParser
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import typing
import langchain_core
from langgraph.graph import StateGraph, END
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser,OutputFunctionsParser
import operator
import json
from langgraph.graph.message import add_messages
from tqdm import tqdm
from utils import *
from langchain import hub
from langchain_community.callbacks import get_openai_callback
from langchain.agents import create_react_agent
from langgraph.graph import StateGraph, END
from langchain.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field, ValidationError
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from tools import *
from parsing_utils import *
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import OpenAI

OPENAI_API_KEY ="" #enter your openai api key here
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm_model_exp = ChatOpenAI(model = "gpt-4o")
llm_model_cheap = ChatOpenAI(model = "gpt-4o-mini")

class GraphState(TypedDict):

    printer_url: str
  
    information_required: Annotated[list, add_messages]
    information_known: Annotated[list, add_messages]
    

    reasoning : List[str]
    adapted_recon_reasoning : Annotated[Sequence[BaseMessage], operator.add]

    solution_reasoning : List[str]
    adapter_solution_reasoning : Annotated[Sequence[BaseMessage], operator.add]
    
    solution_steps : Annotated[list, add_messages]

    observations: str

    docs: dict
    dump:List[str]

    printer_obj: dict
    gcode_commands: dict

    internal_messages: Annotated[Sequence[BaseMessage], operator.add]

    scratchpad: str
    members: List[str]
    
    next: str
    wait_time : int 

    potential_causes: List[str]
    previous_solution: List[str]

members = ["info_planner", "resume_print","recon_executor","sol_planner","sol_executor"]
system_prompt = (
    "You are a 3D printing supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond when the task is done. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members+["FINISH"]
print("Options:",options)
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="internal_messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4o")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

def resume_print_node(state):
    out= resume_print.invoke(state)
    return {"internal_messages":[HumanMessage("Print resumed. You can respond with FINISH now.")]}

def info_reasoning_adapter (model=llm_model_exp, parser = adapt_planner_parser):

    information_gathering_adapt_prompt =  PromptTemplate(
    input_variables=["observations", "reasoning", "info_known"], 
    template= 
        "You are an expert in 3D printing. Based on the given observations and failures in the current layer \nObservations: {observations}\n\
        Your task is to rephrase, rewrite, and reorder each reasoning module to better identify the information needed to resolve these issues in the next layer. Additionally, enhance the reasoning with relevant details to determine which parameters are essential for solving the problem.\n\
        Reasoning Modules: {reasoning}."    
    )
    adapter = information_gathering_adapt_prompt|(model).with_structured_output(parser)
    return adapter

def info_gather_planner(model,parser):

    information_gathering_identification_prompt = PromptTemplate(
    input_variables=["observations","information_known","adapted_recon_reasoning","printer_obj"], 
    template= "You are a 3D printing expert. Given the observed failures: {observations} in the current layer, your task is to think step by step and create a step-by-step plan to operationalize the reasoning modules {adapted_recon_reasoning}. This plan should gather information about the most contributing factors to resolve the failure in the next layer to ensure print rating is 10/10.\n\
    ### Constraints:\n\
    - **No Access to G-Code and Slicer**\n\
    - **Available Objects for Query:** {printer_obj}\n\
    ### Tasks:\n\
    1. **Identify Potential Causes:**\n\
    - Provide a list of potential causes for the observed failure.\n\
    2. **Information Gathering:**\n\
    - Based on the potential causes, specify the information that should be gathered from the printer to narrow down the potential cause.\n\
    - Include information requiring human intervention.\n\
    - Identify the most significant factors causing failures that can be queried by a basic 3D printer running Moonraker.\n\
    - Specify environmental information required and any advanced information that might not be available on basic 3D printers.\n\
    ### Requirements:\n\
    - Ensure the information needed is very specific and focuses on the most probable cause of the failure.\n\
    - Avoid semantically repeated or redundant information.\n\
    ### Provided Information:\n\
    - **Known Information:** {information_known}\n\
    **Note:** Your job is to think step-by-step like an engineer and generate a comprehensive list of required information to facilitate accurate conclusions in similar future tasks. Do not produce invalid content."
    )

    recon = information_gathering_identification_prompt|(model).with_structured_output(parser)

    return recon


def recon_node(state,model,name):
    extracted_info =[]
    tools = [query_printer,query_gCode]
    
    pp = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools',    'information_known','printer_obj'], 
    template='You are tasked with querying a 3D printer running the Moonraker API for requested information.\n\
    You have access to the following tools:\n\n{tools}\n\n\
    Here is the information known to you {information_known}.\n\
    NOTE: The current maximum print speed is (speed*speed_factor) found in gcode_move.\n\
    Here are the printer and Gcode objects you can query {printer_obj}, use your experience in determining which object to call to get current information.\n You can write additional python code to interpret the information or process information from the printer.\n If you dont find the required info try semantically similar words.\n\
    Use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}')

    
    agent = create_react_agent(llm_model_exp,tools=tools, prompt=pp)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True,handle_parsing_errors=True)
    print("info req:\n", state["information_required"])

    for i in range(len(state["information_required"][0].content)):

        print(f"\n\nGathering information from printer for "+ str((state["information_required"][0].content)[i])+ "\n\n")

        information_known = state["information_known"]

        agent_out = agent_executor.invoke({"input":(state["information_required"][0].content)[i],"information_known":information_known, "printer_obj":state["printer_obj"]})

        if agent_out["output"] != "Agent stopped due to iteration limit or time limit":

            print("agent be good") 
            extracted_info.append(agent_out["output"])

        print("\n Information Known: ",(state["information_known"]))

    return {"information_known":extracted_info,"internal_messages":[HumanMessage("Information gathered from the printer")]}

def solution_reasoning_adapter(model,parser=adapt_solver):
    solution_adapt_prompt =  PromptTemplate(
    input_variables=["observations", "solution_reasoning", "information_known"], 
    # template= 
    #     "You are an expert at 3D printing. This is the information known to you {information_known}\n\
    #     Given the failures : {observations} in the current layer number, you must select relevant reasoning modules, reorder them and rephrase selected reasoning module based on the specific observations and failures to better identifies the information required to solving the failures in the next layer.\
    #     Reasoning Modules: {solution_reasoning}.\n\
    #     You can add more reasoning modules and more information to the reasoning modules if you think it will help in automated solving of the failures without human involvement or gathering more information. Do not produce invalid content."    
    # )
    template= "You are an expert in 3D printing. Here is the information known to you: {information_known}.\n\
    Given the failures observed in the current layer number: {observations}, your task is to select the relevant reasoning modules, reorder them, and rephrase the selected reasoning modules based on the specific observations and failures. Your goal is to provide the reasoning required to generate a solution plan for addressing the failures in the next layer.\n\
    Reasoning Modules: {solution_reasoning}.\n\
    You can add more reasoning modules and include additional information if you believe it will aid in the automated resolution of failures without human involvement or in gathering more information. Do not produce invalid content.\
    ")
    adapter = solution_adapt_prompt|(model).with_structured_output(parser)
    return adapter

def solution_planner(model,parser=solution_planner_parser):

    solution_planner_prompt = PromptTemplate(
    input_variables=["observations","information_known","adapter_solution_reasoning", "gcode_commands","previous_solution"], 
    # template= " You are an expert at 3D printing. Your task is to solve the failure by choosing best parameters and parameter values such that the print rating is 10. Given the failures : {observations} in the current layer and previously attempted solution {previous_solution}.\n You must operationalize the reasoning modules {adapter_solution_reasoning} into a step-by-step plan to solve the failure in the next layer by changing the print parameters or running commands via Moonraker api. \n\
    # Commands: {gcode_commands}\
    #  \nYou donot have access to G-Code of the part and Slicer\
    # \nYou need to provide solutions that requires human intervention and solution plan which includes parameters and the best possible values that can be changed autonomously on the printer to solve the problem in the next layer. Based on known information choose the best parameter values to solve the problem in the next segment/layer.\n\
    # NOTE: The print speed can only be changed by speed_factor\n\
    # Ensure the generated solution is very specific and not semantically repeated, also provide a rationale for the generated solution steps in the same line.\
    # \nCurrently the printer is paused.\
    # \n Here is the information known to you: {information_known}.\n\
    # \n\nNote: Your job is to generate a detailed step by step solution list so that in the future you can implement it on the printer to solve the failure in the next layer. Do not produce invalid content. Donot repeat the same/semantically similar solution steps. Do not resume print."    
    # )
    template="You are an expert at 3D printing.\n\
    Here is the information known to you: {information_known}.\n\
    Your task is to solve the failure by choosing the best parameters and parameter values such that the print rating is 10. Given the failures observed in the current layer: {observations}, and the previously attempted solution: {previous_solution}, you must operationalize the reasoning modules {adapter_solution_reasoning} into a step-by-step plan to solve the failure in the next layer by changing the print parameters or running commands via the Moonraker API.\n\
    Commands: {gcode_commands}\n\
    You do not have access to the G-Code of the part and the Slicer.\n\
    You need to provide solutions that require minimal human intervention and a solution plan that includes parameters and the best possible values that can be changed autonomously on the printer to solve the problem in the next layer.\n\
    NOTE: The print speed can only be changed by adjusting the speed_factor.\n Donot increase extruder temperature beyond 205 deg C\
    Ensure the generated solution is very specific and not semantically repetitive, and provide a rationale for the generated solution steps in the same line.\n\
    Currently, the printer is paused.\n\
    Note: Your job is to generate a detailed step-by-step solution list so that in the future you can implement it on the printer to solve the failure in the next layer. Do not produce invalid content. Do not repeat the same or semantically similar solution steps. Do not resume print.\
")

    recon = solution_planner_prompt|(model).with_structured_output(parser)
    return recon


def create_solution_planner_node(state, agent, adapter, name):
    adapter_out = adapter.invoke(state)
    # node_out = {"adapter_solution_reasoning":[HumanMessage(adapter_out.adapted_prompts)]}
    state["adapter_solution_reasoning"].extend(adapter_out.adapted_prompts)
    agent_out = agent.invoke(state)
    node_out = {"solution_steps":[HumanMessage(agent_out.step_commands_to_run)],"internal_messages":[HumanMessage("Solution plan generated, go to next step")]}
    return node_out


def create_info_gather_node(state,agent,adapter,name):
    adapter_out = adapter.invoke(state)
    # node_out = {"adapted_recon_reasoning":[HumanMessage(adapter_out.adapted_prompts)]}
    state["adapted_recon_reasoning"].extend(adapter_out.adapted_prompts)
    agent_out = agent.invoke(state)
    node_out = {"information_required":[HumanMessage(agent_out.information_required_from_printer)],"internal_messages":[HumanMessage("Plan generated for information gathering, go to next step")],
    "potential_causes":[HumanMessage(str(agent_out.potential_causes))]}
    print("Node out done")
    return node_out


def solution_executor_node(state,model,name):
    
    pp2 = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools',    'information_known','gcode_commands'], 
    template='You are tasked with querying a 3D printer running the Moonraker API for requested information.\n\
    You have access to the following tools:\n\n{tools}\n\n\
    Here is the information known to you {information_known}.\n\
    Here are the printer parameters you can modify with gcode commands: {gcode_commands}, use your experience in determining which command to execute to implement the solution plan.\n If you dont find the required command to run try semantically similar words.\n\
    Use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}')


    tools = [change_parameters]
    agent = create_react_agent(llm_model_exp, tools, prompt=pp2)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True,handle_parsing_errors=True)

    for i in range(len(state["solution_steps"][0].content)):

        print(f"\n\nSetting value "+ str((state["solution_steps"][0].content)[i])+ "\n\n")

        information_known = state["information_known"]

        agent_out = agent_executor.invoke({"input":(state["solution_steps"][0].content)[i],"information_known":information_known, "gcode_commands":state["gcode_commands"]})

        if agent_out["output"] != "Agent stopped due to iteration limit or time limit":

            print("agent be good") 
            # extracted_info.append(agent_out["output"])

            

        # print("\n Information Known: ",(state["information_known"]))
    
    # state["information_known"]
    state["internal_messages"].extend([HumanMessage("Parameters changed on the printer, go to resume print")])
    return state


def get_graph():

    info_adapter = info_reasoning_adapter(llm_model_exp)
    info_planner_agent = info_gather_planner(llm_model_exp,parser=recon)
    info_planner_node = functools.partial(create_info_gather_node,agent=info_planner_agent,adapter=info_adapter,name="info_planner")
    recon_executor_node = functools.partial(recon_node,model=llm_model_exp,name="recon_executor")

    solution_adapter = solution_reasoning_adapter(llm_model_exp)
    solution_planner_r = solution_planner(llm_model_exp,parser=solution_planner_parser)
    solution_planner_node = functools.partial(create_solution_planner_node,agent=solution_planner_r,adapter=solution_adapter,name="sol_planner")

    solution_executor = functools.partial(solution_executor_node,model=llm_model_exp,name="sol_executor")


    workflow = StateGraph(GraphState)
    workflow.add_node("supervisor", supervisor_chain)
    workflow.add_node("info_planner", info_planner_node)
    workflow.add_node("recon_executor", recon_executor_node)
    workflow.add_node("sol_planner", solution_planner_node)
    workflow.add_node("sol_executor", solution_executor)
    workflow.add_node("resume_print", resume_print_node)

    for member in members:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("supervisor")

    graph= workflow.compile()
    return graph


# g = get_graph()

# reasoning_planner=load_text_file("./prompts/info_reasoning.txt")
# observation=load_text_file("failure.txt")
# printer_objects = load_text_file("./prompts/printer_objects.txt")
# solution_reasoning= load_text_file("./prompts/solution_reasoning.txt")
# gcode_cmd = load_text_file("./prompts/gcode_commands.txt")
# out = g.invoke(
#     {
#                             "internal_messages": ["Given the failure (if any) plan for what information is required to identify the issue, query printer for the required information, plan the solution steps to solve the problem, execute the solution plan and then resume print. If no problem is detected, resume print."],

#                             "printer_url": "http://172.26.79.51",

#                             "information_known":["Printer url is http://172.26.79.51","Filament type is PLA","printer model is creatlity ender 5 plus", f"Printer status is paused", "Current layer is {1}", "Tool position is at the home position", "BED is perfectly Calibrated"],

#                             "observations": observation,

#                             "reasoning": reasoning_planner,
#                             "solution_reasoning": solution_reasoning, 
#                             "printer_obj": printer_objects,
#                             "adapted_recon_reasoning" : [],
#                             "adapter_solution_reasoning" : [],
#                             "gcode_commands":gcode_cmd,
#                         },

# )

# #save output to file text
# with open("output.txt", "w") as text_file:
#     text_file.write(str(out))