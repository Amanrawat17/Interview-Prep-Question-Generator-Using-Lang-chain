import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

# Set your API key for LangChain's integration with Google Generative AI
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBmE2thQmqHxhtiINEJNNbDiumgEKkqHoc' 

class InterviewRequest(BaseModel):
    role: str = Field(..., min_length=1, max_length=100)

    @field_validator('role')
    def validate_role(cls, v):
        if not v.strip():
            raise ValueError('Role cannot be empty.')
        return v

def generate_interview_questions(role):
    # LangChain model initialization
    llm = ChatGoogleGenerativeAI(model='gemini-pro', api_key=os.environ['GOOGLE_API_KEY'])
    
    # Prompt for generating interview questions
    prompt = f"Generate 5 interview questions for a {role} role. Provide hints and model answers for each question."
    
    try:
        response = llm.invoke(prompt)
        # Check if response has content
        if not response.content:
            raise ValueError("No content returned from the model.")
        
        # Extract questions and answers from the response
        questions = parse_questions(response.content)
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def parse_questions(data):
    questions = []
    lines = data.split('\n')
    current_question = {}
    
    for line in lines:
        if line.startswith("Q:"):
            if current_question:
                questions.append(current_question)
            current_question = {"question": line[2:].strip(), "hint": "", "answer": ""}
        elif line.startswith("Hint:"):
            current_question["hint"] = line[5:].strip()
        elif line.startswith("Answer:"):
            current_question["answer"] = line[7:].strip()
    
    if current_question:
        questions.append(current_question)
        
    return questions

def main():
    st.title("Interview Prep Question Generator")
    st.header("Get Ready for Your Next Interview! 🎯")
    
    role_input = st.text_input("Enter Job Role or Domain (e.g., Data Scientist, Marketing Manager)")
    
    if st.button("Generate Questions"):
        try:
            # Validate input using Pydantic
            request = InterviewRequest(role=role_input)
            st.success("Generating interview questions...")
            questions = generate_interview_questions(request.role)
            
            st.subheader("Interview Questions")
            for idx, q in enumerate(questions, 1):
                st.write(f"**Q{idx}:** {q['question']}")
                if st.button(f"Show Hint for Q{idx}"):
                    st.info(f"Hint: {q['hint']}")
                if st.button(f"Show Answer for Q{idx}"):
                    st.success(f"Answer: {q['answer']}")
        
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
