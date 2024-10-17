import openai
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

# Get your API credentials from environment variables
EDAMAM_APP_ID = os.getenv("EDAMAM_APP_ID")
EDAMAM_APP_KEY = os.getenv("EDAMAM_APP_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Base URL for Edamam API
BASE_URL = "https://api.edamam.com/api/nutrition-details"

def nutrition_api(recipe_data: dict) -> dict:
    """
    Submit a recipe to Edamam Nutrition API and get nutrition analysis.
    
    :param recipe_data: A dictionary containing the recipe details.
    :return: The nutrition analysis response in JSON format.
    """
    # Construct the full URL with app_id and app_key as query parameters
    url = f"{BASE_URL}?app_id={EDAMAM_APP_ID}&app_key={EDAMAM_APP_KEY}"
    
    # Set the headers for the POST request
    headers = {
        "Content-Type": "application/json"
    }

    # Make the POST request to the Edamam API with the recipe data
    response = requests.post(url, json=recipe_data, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON response containing the nutrition analysis
        return response.json()
    else:
        # If there's an error, raise an exception with the error message
        raise Exception(f"Failed to get nutrition data: {response.status_code}, {response.text}")

def get_completion(messages, model="gpt-4.o", temperature=0, max_tokens=300, tools=None):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools
    )
    return response.choices[0].message

def get_nutrition_info(recipe_text: str):
    """
    Pass a recipe into this function to extract ingredients and get nutrition info.
    
    :param recipe_text: Recipe provided by the user in natural language.
    :return: Nutrition analysis result from Edamam API.
    """
    # Define the function (tool) to handle extracting ingredients and nutrition analysis
    functions = [
        {
            "name": "nutrition_api",
            "description": "Fetches nutrition information for a given recipe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_data": {
                        "type": "object",
                        "description": "A dictionary containing the recipe details, including ingredients.",
                        "properties": {
                            "title": {"type": "string"},
                            "ingr": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of ingredients."
                            }
                        },
                        "required": ["ingr"]
                    }
                },
                "required": ["recipe_data"],
            }
        }
    ]

    # Example user input asking for nutrition info from a recipe
    user_message = f"Can you analyze the nutrition of this recipe: {recipe_text}"
    messages = [
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Get the initial response with function calling setup
    response = get_completion(messages, functions=functions)

    # Parse and execute the function call(s) if required
    function_map = {
        "nutrition_api": nutrition_api  # Map the tool name to the Python function
    }

    # If the model suggests calling a function, execute it
    if 'function_call' in response:
        function_call = response['function_call']
        function_name = function_call['name']
        function_args = eval(function_call['arguments'])
        
        # Call the corresponding function with the extracted arguments
        tool_response = function_map[function_name](**function_args)

        # Return the nutrition info received from Edamam API
        return tool_response

    # Return if no function call is made
    return response