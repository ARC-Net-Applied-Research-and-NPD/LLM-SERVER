from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os 
import json 
from langchain_openai import ChatOpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

with open('json/transcription_output.json', 'r') as file:
    transcription_output = json.load(file)

with open('json/evaluation.json', 'r') as file:
    evaluation = json.load(file)
    
with open('json/output.json', 'r') as file:
    audio_metrics = json.load(file)

def score_analyser(transcription, metrics, evaluation):
    model = ChatOpenAI(model=MODEL, api_key=api_key)
    output_parser = JsonOutputParser()
    prompt_template = ChatPromptTemplate.from_messages([
                ("system", 
                "You are an interview scorer, you will be provided with the interviewee's video transcript, metrics of the interviewee's audio and video, and the interviewer's evaluation. Your task is to score the candidate's performance based on these inputs. The scoring scale is: Needs Improvement -> Poor -> Satisfactory -> Good -> Excellent. Provide the score in JSON format, like question1: <score>, question2: <score>, etc., for all questions provided, with no extra commentary or explanation."),
                ("user", 
                """
    Transcript: {transcript}
    Audio and Video Metrics: {metrics}
    Interviewer's Evaluation: {evaluation}
                """
                )
            ])
    chain = prompt_template | model | output_parser
    output = chain.invoke({
        'transcript': transcription,
        'metrics': metrics,
        'evaluation': evaluation
    })
    return output 

print(score_analyser(transcription_output, audio_metrics, evaluation))