import os
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import instructor
import logging
from datetime import datetime
from typing import Optional

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
class EventExtraction(BaseModel):
    description: str = Field(description="Raw description of the event")
    is_calendar_event:bool=Field(description="Whether the event is a calendar event")
    conf_score:float = Field(description="Confidence score between 0 and 1")

class EventDetails(BaseModel):
    name:str = Field(description="Name of the event")
    date:str = Field(description="Date and time of the event. Use ISO 8601 to format this value.")
    duration:int = Field(description="Duration of the event in minutes")
    participants:list[str] = Field(description="List of participants in the event")

class EventConfirmation(BaseModel):
    confirmation:str = Field(description="Confirmation message for the event")
    calendar_link:Optional[str] = Field(description="Link to the calendar event, if available")
#===================================


#Define the functions
#===================================
def extract_event_details(user_ip : str) -> EventExtraction:
    logger.info('Starting event extraction analysis')
    logger.debug(f'User input: {user_ip}')

    today = datetime.today()
    date_ctx = f"Today is {today.strftime('%A, %B %d, %Y')}."

    output = client.chat.completions.create(
        messages=[
                {
                    "role": "system",
                    "content": f"{date_ctx} Analyze if the text describes a calendar event.",
                },
                {"role": "user", "content": user_ip},
            ],
        response_model=EventExtraction,
        model="llama-3.3-70b-versatile",
    )
    logger.info(
        f"Extraction complete - Is calendar event: {output.is_calendar_event}, Confidence: {output.conf_score:.2f}"
    )
    return output


# print(extract_event_details("Meeting with John Doe on 2022-12-15 at 10:00 AM"))


def parse_event_details(description:str) -> EventDetails:
    logger.info('Parsing event details')
    logger.debug(f'Event description: {description}')


    today = datetime.now()
    date_ctx = f"Today is {today.strftime('%A, %B %d, %Y')}."

    output = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"{date_ctx} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.",
            },
            {"role": "user", "content": description},
        ],
        response_model=EventDetails,
        model="llama-3.3-70b-versatile",
    )
    logger.info(
        f"Parsed event details - Name: {output.name}, Date: {output.date}, Duration: {output.duration}min"
    )
    logger.debug(f"Participants: {', '.join(output.participants)}")
    return output

# print(parse_event_details("Meeting with John Doe on 2022-12-15 at 10:00 AM"))

def generate_confirmation(event: EventDetails) -> EventConfirmation:
    logger.info("Generating event confirmation")

    output = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign of with your name; Susie",
            },
            {"role": "user", "content": str(event.model_dump())},
        ],
        response_model=EventConfirmation,
        model="llama-3.3-70b-versatile",
    )
    logger.info("Confirmation message generated successfully")
    return output

# print(generate_confirmation(parse_event_details("Meeting with John Doe on 2022-12-15 at 10:00 AM")))
#===================================



#chain the functions
#===================================
def process_request(usr_ip: str) -> Optional[EventConfirmation]:
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {usr_ip}")

    first_output = extract_event_details(usr_ip)

    if (
        not first_output.is_calendar_event
        or first_output.conf_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {first_output.is_calendar_event}, confidence: {first_output.conf_score:.2f}"
        )
        return None
    
    logger.info("Gate check passed, proceeding with event processing")

    event_details = parse_event_details(first_output.description)
    confirmation = generate_confirmation(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation
#================================

#Test the function
#================================
# user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."
user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"
result = process_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")
#================================
