from flask import Flask, request, jsonify, render_template
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from geopy.geocoders import Nominatim
import os 
from dotenv import load_dotenv  
import json 

app = Flask(__name__)

load_dotenv()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getdisasterDataFromIdea', methods=['POST'])
def get_disaster_data_from_idea():
    try:
        # Get the disaster idea from the request body
        data = request.get_json()
        disasterIdea = data.get('idea')

        llm = OpenAI(temperature=0.9, openai_api_key=os.environ.get('OPEN_AI_KEY'))
        params = {
            "engine": "google",
            "gl": "us",
            "hl": "en",
            # "domain": ["ndtv.com", "bbc.in", "thehindu.com"],
        }
        search = SerpAPIWrapper(params=params)
        tool = Tool(
            name="search_tool",
            description="To search for relevant information about the disaster",
            func=search.run,
        )
        agent = initialize_agent([tool], llm, agent="zero-shot-react-description", verbose=True)

        prompt = f"The user is interested in finding locations affected by disasters. Given the prompt {disasterIdea}, find the relevant affected areas of disaster (like specific landmarks, attractions, or sites) separated by commas."
        response = agent.run(prompt)

        if "Google hasn't returned any results for this query" in response:
            return jsonify({'error': 'No results found'})

        locations = [place.strip() for place in response.split(',')]

        geolocator = Nominatim(user_agent="DisasterWatch")
        locations_data = []
        for location in locations:
            # Remove "district" from location name
            location = location.replace(" district", "")
            try:
                geo_location = geolocator.geocode(location)
                if geo_location is not None:
                    locations_data.append({
                        'location': location,
                        'latitude': geo_location.latitude,
                        'longitude': geo_location.longitude,
                    })
                else:
                    app.logger.info(f"Location not found or geocoding error: {location}")
            except Exception as e:
                app.logger.info(f"An error occurred: {str(e)}")

        response_schemas = [
            ResponseSchema(name="commentary", description="news about the disaster"),
            ResponseSchema(name="date", description="date of the disaster"),
            ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        question = f"The user is interested in finding disaster commentary for the location {location}. Find the latest news about the disasters along with the date and the location of the disaster, along with the source (link) of the news"
        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("Answer the user's question as best as possible.\n{format_instructions}\n{question}")
            ],
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )

        input = prompt.format_prompt(question=question)

        commentary = []
        dates = []
        sources = []
        for location_data in locations_data:
            location = location_data['location']
            prompt = f"The user is interested in finding disaster commentary for the location {location}. Find the latest news about the disasters along with the date and the location of the disaster, along with the source of the news"
            response = agent.run(input)

            try:
                response_data = json.loads(response)  # Parse the JSON response into a dictionary
                commentary.append(response_data['answer'])
                dates.append(response_data['date'])
                sources.append(response_data['source'])
            except json.JSONDecodeError as json_error:
                app.logger.error(f"JSON parsing error: {json_error}")
                commentary.append("Error parsing response")
                dates.append("")
                sources.append("")

        # Store data in a JSON file
        data_to_store = {
            'dates': dates,
            'locations': locations,
            'commentary': commentary,
            'sources': sources
        }

        with open('disaster_data.json', 'w') as json_file:
            json.dump(data_to_store, json_file)

        # Return data to the client
        return jsonify({'dates': dates, 'locations': locations, 'commentary': commentary, 'sources': sources})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.debug = True
    app.run(port=3000)
