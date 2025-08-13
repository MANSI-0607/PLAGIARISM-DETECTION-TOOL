import streamlit as st
import string
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.corpus import stopwords
from itertools import chain

# Ensure resources are available
for resource in ['brown', 'punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

@st.cache_data
def get_stopwords():
    return set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    sw = get_stopwords()
    return [t for t in tokens if t.isalpha() and t not in sw]

@st.cache_resource
def train_unigram_model():
    tokens = nltk.corpus.brown.words()
    train_data, padded_vocab = padded_everygram_pipeline(1, tokens)
    model = MLE(1)
    model.fit(train_data, padded_vocab)
    return model

def plot_most_common_words(text):
    tokens = preprocess_text(text)
    word_freq = nltk.FreqDist(tokens)
    most_common_words = word_freq.most_common(10)

    if most_common_words:
        words, counts = zip(*most_common_words)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(words, counts)
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Most Common Words')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("No words to display.")

def plot_repeated_words(text):
    tokens = preprocess_text(text)
    word_freq = nltk.FreqDist(tokens)
    repeated_words = [word for word, count in word_freq.items() if count > 1][:10]

    if repeated_words:
        words, counts = zip(*[(word, word_freq[word]) for word in repeated_words])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(words, counts)
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Repeated Words')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("No repeated words found.")

def calculate_perplexity(text, model):
    tokens = preprocess_text(text)
    test_data, _ = padded_everygram_pipeline(model.order, tokens)
    flat_ngrams = chain.from_iterable(test_data)
    return model.perplexity(flat_ngrams)

def calculate_burstiness(text):
    tokens = preprocess_text(text)
    if not tokens:
        return 0.0
    freq_dist = nltk.FreqDist(tokens)
    freqs = list(freq_dist.values())
    avg = sum(freqs) / len(freqs)
    variance = sum((f - avg) ** 2 for f in freqs) / len(freqs)
    return variance / (avg ** 2)


def is_generated_text(perplexity, burstiness_score):
    disclaimer = """
    <div style='padding:10px;border-radius:8px;background-color:#fce4ec;color:#c2185b;'>
    <b>DISCLAIMER:</b> This tool does NOT detect plagiarism or definitively prove authorship. 
    It only provides an approximate indicator based on perplexity and burstiness patterns. 
    Use results responsibly.
    </div>
    """
    score = max(0, 100 - perplexity) * (1 - min(burstiness_score, 1))
    if score > 50:
        st.error(f"Likely AI-generated (score {score:.1f})")
    else:
        st.success(f"Likely human-written (score {score:.1f})")

    return disclaimer




def main():
    st.title("Language Model Text Analysis")
    text = st.text_area("Enter the text you want to analyze", height=200)

    if st.button("Analyze"):
        if text.strip():
            model = train_unigram_model()
            perplexity = calculate_perplexity(text, model)
            burstiness_score = calculate_burstiness(text)

            col1, col2 = st.columns(2)
            col1.metric("Perplexity", f"{perplexity:.2f}")
            col2.metric("Burstiness", f"{burstiness_score:.2f}")

            disclaimer= is_generated_text(perplexity, burstiness_score)
            st.markdown(disclaimer, unsafe_allow_html=True)


            st.subheader("Word Frequency Analysis")
            plot_most_common_words(text)
            plot_repeated_words(text)
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
