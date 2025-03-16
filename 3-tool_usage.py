import os
from groq import Groq
from dotenv import load_dotenv
import requests, json

# Load the API key from the .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=os.environ.get(GROQ_API_KEY),
)

#Define the tool
def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

#function to call the weather function
def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)

# Define the available tools (i.e. functions) for our model to use
tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for provided coordinates in celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                    "additionalProperties": False,
                },
            "strict": True,
            },
        }
    ]

#LLM Calling
system_prompt = "You are a helpful weather assistant."
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Lahore today?"},
]
#Getting the first response
response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # LLM to use
        messages=messages, # Conversation history
        stream=False,
        tools=tools, # Available tools (i.e. functions) for our LLM to use
        tool_choice="auto", # Let our LLM decide when to use tools
        max_completion_tokens=4096 # Maximum number of tokens to allow in our response
    )

# print(response.choices[0].message.content)

#Checking the tool calls
response_message = response.choices[0].message
tool_calls = response_message.tool_calls
if tool_calls:
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        # Call the tool and get the response
        result = call_function(function_name, function_args)
        print("RESULT: ",result)
        messages.append(
                {
                    "tool_call_id": tool_call.id, 
                    "role": "tool", # Indicates this message is from tool use
                    "name": function_name,
                    "content": json.dumps(result),
                }
            )
        print("RESULT_MSG",messages)


# Make a second API call with the updated conversation
second_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
# Return the final response
print("Final Response: ",second_response.choices[0].message.content)