import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from dotenv import load_dotenv
import os
from prompt import example

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Personalized Cover Letter Generator",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Personalized Cover Letter Generator")
st.sidebar.markdown("### Welcome to the Lyzr Personalized Cover Letter Generator!")
st.sidebar.markdown("This app harnesses power of Lyzr Automata to Create Personalized Cover letter. You have to Enter Job title,Experience,Sills and company name and it will generate personalized cover letter for you.")
st.markdown("This app uses Lyzr Automata Agent to Generate Cover letter based on your Job title,Experience,Skills and company.")

open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

title = st.sidebar.text_input("Enter Job Title", placeholder="Python Developer")
experience = st.sidebar.slider("Select your Experience", 0, 30, 2)
skill = st.sidebar.text_input("Enter Your Skills", placeholder="Python,Django,AWS")
company = st.sidebar.text_input("Enter Company Name", placeholder="Google")


def cover_letter(job_title,experience,skills,company_name):

    toxicity_agent = Agent(
            role="Cover Letter expert",
            prompt_persona=f"You are an Expert Cover Letter writer.Your Task Is to create cover letter for given details."
        )

    prompt = f"""Write a cover letter for a job application for {job_title} with {experience} years of experience at {company_name}.Have a experience in {skills} Make sure to highlight technical skills, past experiences, and explain why you're passionate about this role and company.
        Follow Below Instructions:
        1/ Cover letter content is upto 300 words not more than it.
        2/ Do not write about Job Posting or advertised Job
        3/ Keep Cover letter very simple and include all skills which is needed for job title.
        4/ Analyse company on internet and also write something about company.

        Example:
        {example}  
        """

    toxicity_task = Task(
        name="Generate Cover Letter",
        model=open_ai_text_completion_model,
        agent=toxicity_agent,
        instructions=prompt,
    )

    output = LinearSyncPipeline(
        name="Cover Letter Pipline",
        completion_message="Cover Letter Generated!!",
        tasks=[
              toxicity_task
        ],
    ).run()

    answer = output[0]['task_output']

    return answer


if st.sidebar.button("Generate Cover Letter",type="primary"):
    solution = cover_letter(title, experience, skill, company)
    st.markdown(solution)

