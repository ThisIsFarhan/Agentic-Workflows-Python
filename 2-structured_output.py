import os
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv
import instructor

# Load the API key from the .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create an instructor client
client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)

# Define the model for the structured output
class UserInfo(BaseModel):
    name:str
    age:int
    email:str

# Define the input text
text = """
John Doe, a 35-year-old software engineer from New York, has been working with large language models for several years.
His email address is johndoe123@example.com.
"""

# Call the completions endpoint with the structured output model
output = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "Your job is to extract user information from the given text."},
        {"role": "user", "content": text}
    ],
    response_model=UserInfo,
    model="llama-3.3-70b-versatile",
)

# Print the structured output
print(f"Name: {output.name}")
print(f"Age: {output.age}")
print(f"Email: {output.email}")