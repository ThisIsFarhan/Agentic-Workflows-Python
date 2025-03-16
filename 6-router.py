import os
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import instructor
import logging
from datetime import datetime
from typing import Optional, Literal

#Setting up logging
#===================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
#===================================


#Load the API key from the .env file and create an instructor client
#===================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)
#===================================

#Define the models for the structured output
#===================================
class ReqType(BaseModel):
    req_type: Literal["new_event","modify_event","other"] = Field(description="Type of request")
    conf_score: float = Field(description="Confidence score between 0 and 1")
    description: str = Field(description="Raw description of the request")

class NewEventDetails(BaseModel):
    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: int = Field(description="Duration in minutes")
    participants: list[str] = Field(description="List of participants")

class Change(BaseModel):
    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")

class ModifyEventDetails(BaseModel):
    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(description="List of changes to make")
    participants_to_add: list[str] = Field(description="New participants to add")
    participants_to_remove: list[str] = Field(description="Participants to remove")

class CalendarResponse(BaseModel):
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if applicable")
#===================================

#Define the functions
#===================================
def router_req(usr_ip:str) -> ReqType:
    logger.info("Routing Request")
    output = client.chat.completions.create(
        messages=[
                {
                    "role": "system",
                    "content": f"Determine if this is a request to create a new calendar event or modify an existing one.",
                },
                {"role": "user", "content": usr_ip},
            ],
        response_model=ReqType,
        model="llama-3.3-70b-versatile",
    )
    logger.info(
        f"Request routed as: {output.req_type} with confidence: {output.conf_score}"
    )
    return output


def new_event(desc:str) -> CalendarResponse:
    logger.info("Processing new event request")

    output = client.chat.completions.create(
        messages=[
                {
                    "role": "system",
                    "content": f"Extract details for creating a new calendar event.",
                },
                {"role": "user", "content": desc},
            ],
        response_model=NewEventDetails,
        model="llama-3.3-70b-versatile",
    )
    logger.info(
        f"New event: {output.model_dump_json(indent=2)}"
    )
    return CalendarResponse(
        success=True,
        message=f"Created new event '{output.name}' for {output.date} with {', '.join(output.participants)}",
        calendar_link=f"calendar://new?event={output.name}"
    )

def modify_event(desc:str) -> CalendarResponse:
    logger.info("Processing modify event request")

    output = client.chat.completions.create(
        messages=[
                {
                    "role": "system",
                    "content": f"Extract details for modifying an existing calendar event.",
                },
                {"role": "user", "content": desc},
            ],
        response_model=ModifyEventDetails,
        model="llama-3.3-70b-versatile",
    )
    logger.info(
        f"Modify event: {output.model_dump_json(indent=2)}"
    )
    return CalendarResponse(
        success=True,
        message=f"Modified event '{output.event_identifier}' with required changes",
        calendar_link=f"calendar://modify?event={output.event_identifier}"
    )
#===================================

#Chain the functions
#===================================
def process_request(usr_ip: str) -> Optional[CalendarResponse]:
    logger.info("Processing calendar request")

    route_result = router_req(usr_ip)
    if route_result.conf_score < 0.7:
        logger.warning(f"Low confidence score: {route_result.confidence_score}")
        return None
    
    if route_result.req_type == "new_event":
        return new_event(route_result.description)
    elif route_result.req_type == "modify_event":
        return modify_event(route_result.description)
    else:
        logger.warning("Request type not recognized")
        return None
#================================


#Test the function
#================================
# new_event_input = "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
# result = process_request(new_event_input)
# if result:
#     print(f"Response: {result.message}")

# modify_event_input = (
#     "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?"
# )
# result = process_request(modify_event_input)
# if result:
#     print(f"Response: {result.message}")

invalid_input = "What's the weather like today?"
result = process_request(invalid_input)
if not result:
    print("Request not recognized as a calendar operation")