import os 
from apikey import apikey 

import openai
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi


apikey = 'YOUR API KEY'
openai.api_key = apikey
# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Function to generate questions using OpenAI API
def generate_questions_openai(text, num_questions=3):
    """
    Generate questions from the given text using OpenAI GPT-3.5.
    
    Parameters:
        text (str): The text from which to generate questions.
        num_questions (int): Number of questions to generate.

    Returns:
        list: List of generated questions.
    """
    # Define a prompt to generate questions
    prompt = f"Generate 3 questions based on the following text:\n\n{text}\n\nQuestions:\n"

    # Generate questions using GPT-3.5
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50,  # Adjust the max_tokens as needed
        n=num_questions
    )

    questions = [item['text'].strip() for item in response.choices]
    return questions

def generate_answers_openai(text,questions):
   
    # Define a prompt to generate questions
    prompt1 = f" Provide a Clear and Concise Answer, Support with Evidence, Define Key Terms for these questions {questions} based on text {text} \n"

    # Generate questions using GPT-3.5
    response1 = openai.Completion.create(
        engine="davinci",
        prompt=prompt1,
        max_tokens=50,  # Adjust the max_tokens as needed
    )

    answer = [item['text'].strip() for item in response1.choices]
    return answer

# App framework
st.title('EnthyNotes')
youtube_video = st.text_input('Enter your Lecture- Youtube link') 

# Ensure a valid YouTube video URL is provided
if 'youtube.com' not in youtube_video:
    st.error("Please enter a valid YouTube video URL.")
    st.stop()  # Stop execution if invalid URL

video_id = youtube_video.split("=")[-1]

# Fetch the transcript and handle errors
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
except Exception as e:
    st.error(f"An error occurred while fetching the transcript: {str(e)}")
    st.stop() 

result = ""
for i in transcript:
    result += ' ' + i['text']

summarizer = pipeline('summarization')    

num_iters = int(len(result)/1000)

summarized_text = []
for i in range(0, num_iters + 1):
  start = i * 1000
  end = (i + 1) * 1000
  out = summarizer(result[start:end])
  out = out[0]
  out = out['summary_text']
  summarized_text.append(out)

summarized_text_array = ' '.join(summarized_text)  

notes_template = PromptTemplate(
    input_variables=['summarized_text_array'],
    template='write me detailed notes from the youtube transcript while leveraging with wikipedia research and following these rules: 1. Clear Heading 2. Structured Outline 3. Key Concepts and Definitions 4. Examples and Case Studies 5. Important Formulas and Equations: {summarized_text_array}  '
)

# Memory 
notes_memory = ConversationBufferMemory(memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
notes_chain = LLMChain(llm=llm, prompt=notes_template, verbose=True, output_key='notes', memory=notes_memory)



# Show stuff to the screen if there's a prompt
if summarized_text_array:
    note = notes_chain.run(summarized_text_array)

    st.write(note) 
    

    with st.expander('Notes History'):
        st.info(notes_memory.buffer)


    # Generate questions using OpenAI API
    generated_questions = generate_questions_openai(' '.join(note), num_questions=3)
    

    # Display generated questions when the user clicks the button
    if st.button("Do you want to test yourself?"):
        st.header("Generated Questions:")
        for i, question in enumerate(generated_questions, 1):
            st.write(f"{i}. {question}")
            
    # Generate answers using OpenAI API
    generated_answers = generate_answers_openai(' '.join(note),' '.join(generated_questions))        
            
    if st.button("Show Answers"):
        st.header("Answers")
        for i, answer in enumerate(generated_answers, 1):
            st.write(f"{i}. {answer}")
       


