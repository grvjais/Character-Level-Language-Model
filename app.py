import streamlit as st
import torch
import pandas as pd

# Import everything from your separate model file
from cll_model import ModelConfig, Bigram, CharDataset, InfiniteDataLoader, generate

# ==========================================
# 1. Data Preparation (UI Specific)
# ==========================================

# Use Streamlit's caching so we don't re-read the file on every slider movement
@st.cache_data
def load_and_prep_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        return None, None
    
    words = [w.strip() for w in data.splitlines() if w.strip()]
    if not words: return None, None
    
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)
    dataset = CharDataset(words, chars, max_word_length)
    return dataset, words

# ==========================================
# 2. Streamlit UI
# ==========================================

st.set_page_config(page_title="Bigram Maker", layout="wide")
st.title("Bigram Character-Level Model Setup")

# --- Sidebar Controls ---
st.sidebar.header("Model Parameters")
file_name = st.sidebar.text_input("Dataset File (.txt)", "names.txt")
max_steps = st.sidebar.slider("Training Steps", 500, 10000, 3000, step=500)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
learning_rate = st.sidebar.number_input("Learning Rate", value=0.1, format="%.3f")

dataset, raw_words = load_and_prep_data(file_name)

if dataset is None:
    st.error(f"Could not load '{file_name}'. Make sure it exists in the same directory.")
    st.stop()

# Show a snippet of the data
with st.expander("View Dataset Preview"):
    st.write(f"Loaded **{len(raw_words)}** words. Vocab size: **{dataset.get_vocab_size()}**.")
    st.write(", ".join(raw_words[:20]) + "...")

# --- Session State to hold the trained model ---
if 'model' not in st.session_state:
    st.session_state.model = None

# --- Training Section ---
if st.button("Train Model", type="primary"):
    st.session_state.model = None # Reset model
    
    config = ModelConfig(vocab_size=dataset.get_vocab_size())
    model = Bigram(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = InfiniteDataLoader(dataset, batch_size=batch_size)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    losses = []
    
    # Training Loop within UI
    for step in range(max_steps):
        X, Y = loader.next()
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Update UI every 100 steps
        if step % 100 == 0 or step == max_steps - 1:
            progress = (step + 1) / max_steps
            progress_bar.progress(progress)
            status_text.text(f"Step {step+1}/{max_steps} | Loss: {loss.item():.4f}")
            losses.append(loss.item())
            loss_chart.line_chart(losses)
            
    st.session_state.model = model
    st.success("Training Complete!")

# --- Generation Section ---
st.markdown("---")
st.subheader("Generate New Text")

if st.session_state.model is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_samples = st.number_input("Number of samples", min_value=1, max_value=50, value=10)
        temperature = st.slider("Temperature (Chaos factor)", 0.1, 2.0, 1.0, 0.1)
        top_k_val = st.number_input("Top K (-1 for none)", value=-1)
        generate_btn = st.button("Generate")
        
    with col2:
        if generate_btn:
            with st.spinner("Generating..."):
                top_k = top_k_val if top_k_val != -1 else None
                X_init = torch.zeros(num_samples, 1, dtype=torch.long)
                steps = dataset.get_output_length() - 1 
                
                # Run Generation
                X_samp = generate(st.session_state.model, X_init, steps, 
                                  temperature=temperature, top_k=top_k, do_sample=True)
                
                # Decode and format results
                results = []
                for i in range(X_samp.size(0)):
                    row = X_samp[i, 1:].tolist()
                    crop_index = row.index(0) if 0 in row else len(row)
                    row = row[:crop_index]
                    results.append(dataset.decode(row))
                
                # Display cleanly
                df = pd.DataFrame(results, columns=["Generated Output"])
                st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Train the model first to generate outputs.")