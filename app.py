import streamlit as st
import os
import glob
import re
from llama_cpp import Llama

st.set_page_config(page_title="LLaMA Lab", page_icon="🧪", layout="wide")

st.title("LLaMA Lab 🦙🧪")
st.write("> A lovely 100-LOC script to chat with LLaMA.cpp.")

with st.sidebar:
    model_dir = "~/Downloads/Models/llama/" # Turn it into an argument.

    model_filenames = [os.path.basename(file) for file in glob.glob(os.path.expanduser((os.path.join(model_dir, "*.bin"))))]
    model = st.selectbox("Model", model_filenames)
    model_path = os.path.expanduser(os.path.join(model_dir, model))

    threads = st.slider("Threads", 1, os.cpu_count(), 4)
    context_size = st.slider("Context size", 0, 2048, value=200, help="Maximum number of tokens that cna be processed by the model, including input. Larger context requires more RAM.")

    st.caption("Parameters:")

    temperature = st.slider("Temperature", 0.0, 2.0, value=0.7, help="Controls randomness: Lowering results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.")
    max_tokens = st.slider("Maximum length", 1, 2048, value=256, help="The maximum number of tokens to generate. Only up to 2,048 tokens shared between prompt and completion. The exact limit varies by model. (One token is roughly 4 characters for normal English text.")

    stop_seqs = st.text_input("Stop sequences", placeholder="\\n, ##, User").strip().split(", ")
    stop_seqs = [] if stop_seqs[0] == '' else stop_seqs

    top_p = st.slider("Top P", 0.0, 1.0, value=0.9, help="Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered")
    top_k = st.slider("Top K", 0, 80, value=40, help="Limits the probability mass that the model can choose from at each step of generation. Only the words that cumulatively add up to a probability of p or higher are considered, and the rest are ignored. A higher top p means more diversity and less predictability, while a lower top p means more consistency and less variation.")
    repeat_penalty = st.slider("Repetition penalty", 0.0, 2.0, value=1.1, help="Penalizes words that have already appeared in the output, making them less likely to be generated again. A higher repetition penalty means less repetition and more diversity, while a lower repetition penalty means more repetition and less diversity.")

def extract_placeholders(prompt):
    pattern = r"\{\{(\w+)\}\}"
    placeholders = re.findall(pattern, prompt)
    placeholders_dict = {key: None for key in placeholders}
    return placeholders_dict

def has_none(dictionary):
    for _, value in dictionary.items():
        if value is None or value.strip() == "":
            return True
    return False

def clip(num, min_value, max_value):
   return max(min(num, max_value), min_value)

@st.cache_resource
def load_model(model_path, **kwargs):
    return Llama(model_path=model_path, **kwargs)

# template = st.selectbox("Select a template", ["default"])

prompt = st.text_area("Prompt", placeholder="""## Assistant: Hello! I'm a helpful assistant ready to answer any of your questions! 
## User: {{MESSAGE}}
## Assisant:""")

placeholders = extract_placeholders(prompt)
for key, value in placeholders.items():
    value = st.text_input(f"{key.capitalize()}:", key=key)
    placeholders[key] = value
    prompt = prompt.replace("{{" + key + "}}", value)

if st.button("✨ Run", type="primary", disabled=has_none(placeholders)):
    st.markdown("## Output:")

    progress = st.progress(0, "Initialization...")
    md_box = st.empty()

    llm = load_model(model_path=model_path, n_threads=threads, n_ctx=context_size)

    response = llm(
        prompt,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop_seqs,
        frequency_penalty=repeat_penalty)

    chunk = []
    for resp in response:
        chunk.append(resp["choices"][0]["text"])
        result = "".join(chunk)

        progress_tokens = clip(len(result) / (max_tokens * 4), 0, 1)
        progress.progress(progress_tokens, "Generating...")

        md_box.markdown(result)

    progress.progress(1.0, "Done!")

    st.divider()

    # st.warning(":warning: These timings are currently just a placeholder!\n\nI don't yet know how to fetch this data from LLaMA.cpp, but I find it essential for a locally running LLM, so I don't want to just remove them.")

#     st.markdown("""### Timings:
# - **Load**: 5.85 sec
# - **Sample**: 70.94 ms / 124 runs (0,57 ms per token)
# - **Prompt eval**: 5.67 sec / 49 tokens (115,63 ms per token)
# - **Eval**: 26.24 sec / 123 runs (213,32 ms per token)
# - **Total**: 32 sec""")

    st.markdown("""How well did it go? Is the response factual, relevant, or just complete nonsense?
Check out the model's official inference demos with these same parameters and prompt.

Still having issues? Feel free to open a bug report! Perhaps we didn't quite get things quite right...""")
