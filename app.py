import streamlit as st
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_groq import ChatGroq

st.set_page_config(page_title="Text To Image", page_icon="🎨", layout="centered")
st.title("🎨 AeonRush Text To Image")
st.write("Powered by LangChain & Groq (GPT OSS 120B)")

groq_api_key = st.secrets.get("GROQ_API_KEY")

def load_llm(api_key):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=1.6,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="high",
        stream=False,
        stop=None,
        api_key=api_key
    )

examples = [    
    { # Example 1
        "review": "Create a Dog Image",
        "answer": "https://images.unsplash.com/photo-1543466835-00a7907e9de1"  # Working dog URL
    },
    { # Example 2
        "review": "Generate a Cat image",
        "answer": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba" # Working cat URL
    },
    { # Example 3
        "review": "Generate a Bear image",
        "answer": "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7" # Working bear URL
    }
]

example_prompt = PromptTemplate(
    input_variables=["review", "answer"],
    template="""
Review:
{review}

Answer:
{answer}
"""
)

llm = load_llm(groq_api_key)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,

    prefix="""
You are an expert in Paintings and in Creating Good, attractive and Beautiful images.

Look at the examples below.
""",

    suffix="""
Review:
{review}

Answer:
""",

    input_variables=["review"]
)

chain = few_shot_prompt | llm

user_query = st.text_input("What image would you like to create?", placeholder="e.g., Create an image of a man standing near a bear")

if st.button("Generate Image Link"):
    if user_query.strip():
        with st.spinner("Thinking and fetching best image link..."):
            try:
                response = chain.invoke({"review": user_query})
                output_text = response.content
                
                st.subheader("Result:")
                st.markdown(output_text)
                
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
    else:
        st.warning("Please enter a valid request query.")

