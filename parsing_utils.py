from pydantic import BaseModel
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict


class image_response_parser(BaseModel):
    """Parses the response from the image_inference"""

    observation: str = Field(description="The observation of the image")
    rating: float = Field(description="The rating of the ongoing 3D print")
    other : Optional[str] = Field(description="Other information about the print")
    problems_found : str


class adapt_planner_parser(BaseModel):
    """Information gathering plan"""

    other: Optional[str] = Field(description="other information about the plan")

    adapted_prompts: List[str] = Field(
        description="Adapted and rephrased prompts to better identify the information required to solve the task"
    )
    preamble: Optional[str] = Field(
        description="preamble to the plan"
    )  

class recon(BaseModel):
    """Information gathering plan"""

    other: Optional[str] = Field(description="other information about the plan")

    information_required_from_printer: List[str] = Field(
        description="Most contributing controllable Parameters required from the basic printer to identify the problem and where to find them"
    )

    information_required_from_human: Optional[List[str]] = Field(description="Information required from the human to identify the problem")

    potential_causes: List[str] = Field( description="Potential causes of the problem")

    env_conditions: Optional[List[str]] = Field(description="Environmental information required to identify the problem")

    adv_info: Optional[List[str]] = Field(description="Advanced information required that might not be available on basic 3D printers to identify the problem")

    preamble: Optional[List[str]] = Field(
        description="preamble to the plan"
    )
    potential_solution: Optional[List[str]] = Field(description="Potential solution to the problem without human")


class solution_planner_parser(BaseModel):
    """Information gathering plan"""

    human_solution: Optional[List[str]] = Field(description="Human help in solving the problem")

    step_commands_to_run: List[str] = Field(
        description="Detailed solution plan to be executed autonomously on the printer, Parameters name and value to be changed to solve the problem or the G-code commands to be executed"
    )

    potential_causes: Optional[List[str]] = Field( description="Potential causes of the problem")

    preamble: List[str] = Field(
        description="preamble to the plan"
    )


class adapt_solver(BaseModel):
    """Information gathering plan"""

    # other: str = Field(description="other information about the plan")

    adapted_prompts: List[str] = Field(
        description="Adapted and rephrased prompts to better identify the information required to solve the task"
    )
    preamble: List[str] = Field(
        description="preamble to the plan and other information"
    )  