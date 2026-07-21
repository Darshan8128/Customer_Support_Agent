import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from config import GEMINI_MODEL, EMBEDDING_MODEL

# Import Gemini components for LLM and Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# langchain v1.3+ moved chains/splitters/schema to langchain-classic / langchain-core
try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_core.documents import Document

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

from deep_translator import GoogleTranslator
try:
    import whisper
except ImportError:
    whisper = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

from scipy.io.wavfile import write
import fitz

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


# Load environment variables (ensure .env file is present)
load_dotenv(override=True)
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or st.secrets.get("ELEVENLABS_API_KEY")
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Define paths for persistent storage
VECTOR_STORE_DIR = "vectorstore"
DOC_FILE = "company_docs.txt" # Base document for knowledge
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "company_vectorstore")

def load_vectorstore():
    """
    Loads an existing FAISS vector store or creates a new one from company documents and uploaded files.
    Uses GoogleGenerativeAIEmbeddings.
    """
    # Return existing vectorstore if already loaded and not updated
    if st.session_state.get('vectorstore') is not None and not st.session_state.get('knowledge_updated', False):
        return st.session_state.vectorstore

    # Initialize Gemini embeddings
    try:
        if not GOOGLE_API_KEY:
            st.error("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
            return None
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"ERROR: Failed to initialize GoogleGenerativeAIEmbeddings. Check GOOGLE_API_KEY. Details: {e}")
        return None

    # Attempt to load existing vector store if not explicitly updated
    if os.path.exists(VECTOR_STORE_PATH) and not st.session_state.get('knowledge_updated', False):
        try:
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            st.session_state.vectorstore = vectorstore
            st.session_state.knowledge_updated = False # Reset flag after loading
            return vectorstore
        except Exception as e:
            st.error(f"ERROR: Failed to load existing vectorstore from {VECTOR_STORE_PATH}. Details: {e}. Attempting to rebuild.")
            # If loading fails, proceed to rebuild the vectorstore (fall through)

    # If no existing store or update is needed, create a new one
    try:
        text = ""
        # Read content from the base company document
        if os.path.exists(DOC_FILE):
            with open(DOC_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            st.warning(f"WARNING: {DOC_FILE} not found. Knowledge base will rely only on uploaded files.")

        # Append content from all text files in the 'uploads' directory
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir) and os.path.isdir(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(uploads_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            text += f"\n\n{f.read()}"
                    except Exception as e:
                        print(f"Failed to read {filename}: {e}")

        # Append content from all uploaded files in session state (via UI)
        if 'files' in st.session_state and st.session_state.files:
            for file_info in st.session_state.files:
                text += f"\n\n{file_info['content']}"

        if not text.strip():
            st.error("ERROR: No document content available to build the knowledge base. Please ensure files exist in 'uploads' or 'company_docs.txt', or upload documents via UI.")
            return None

        # Split text into chunks for embeddings
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        if not chunks:
            st.error("ERROR: Text splitter produced no chunks. Document content might be too short or problematic.")
            return None

        # Convert string chunks into Document objects (required by FAISS)
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Create FAISS vector store from documents and save it locally
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)

        st.session_state.vectorstore = vectorstore
        st.session_state.knowledge_updated = False
        st.success("SUCCESS: New vectorstore created and saved successfully.")
        return vectorstore
    except Exception as e:
        st.error(f"ERROR: Error during vectorstore creation/rebuild. Details: {e}")
        return None

def whisper_transcribe(duration=5, samplerate=16000):
    """
    Records audio from the microphone for a specified duration and transcribes it using Whisper.
    """
    if sd is None or whisper is None:
        st.error("Audio recording/transcription packages are not supported or available on this system/deployment.")
        return ""
        
    st.info(f"Recording for {duration} seconds...")
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
    except Exception as e:
        st.error(f"Failed to access recording hardware: {e}")
        return ""
    st.success("Recording finished.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        write(temp_audio_file.name, samplerate, audio)
        audio_filepath = temp_audio_file.name

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_filepath)
        return result['text']
    except Exception as e:
        st.error(f"Error during Whisper transcription: {e}")
        return ""
    finally:
        try:
            os.remove(audio_filepath)
        except Exception:
            pass

def elevenlabs_tts(text, lang="en", voice_name="Rachel"):
    """
    Generates speech from text using ElevenLabs TTS.
    """
    if not ELEVENLABS_API_KEY:
        st.warning("ElevenLabs API key not found. Skipping text-to-speech.")
        return None

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        model_id = "eleven_multilingual_v2" if lang != "en" else "eleven_monolingual_v1"

        target_voice_id = None
        all_voices = client.voices.get_all()

        if all_voices and all_voices.voices:
            for voice_obj in all_voices.voices:
                if isinstance(voice_obj, Voice) and voice_obj.name.lower() == voice_name.lower():
                    target_voice_id = voice_obj.voice_id
                    break

            if not target_voice_id:
                st.warning(f"Voice '{voice_name}' not found. Using the first available voice.")
                if all_voices.voices[0] and isinstance(all_voices.voices[0], Voice):
                    target_voice_id = all_voices.voices[0].voice_id
                else:
                    st.error("No valid fallback voice found from ElevenLabs.")
                    return None
        else:
            st.error("No voices available from ElevenLabs. Please check your ElevenLabs account and API key.")
            return None

        if not target_voice_id:
            st.error("Failed to select a voice for ElevenLabs TTS.")
            return None

        audio_bytes = client.text_to_speech.convert(
            voice_id=target_voice_id,
            text=text,
            model_id=model_id,
        )

        temp_audio_file = tempfile.mktemp(suffix=".mp3")
        with open(temp_audio_file, "wb") as f:
            f.write(audio_bytes)

        return temp_audio_file
    except Exception as e:
        st.error(f"ElevenLabs TTS error: {e}")
        return None

@st.cache_resource
def get_whisper_model():
    if whisper is None:
        return None
    try:
        return whisper.load_model("tiny")
    except Exception:
        try:
            return whisper.load_model("base")
        except Exception:
            return None


def transcribe_audio_bytes(audio_bytes):
    """
    Transcribes raw audio bytes (e.g. from browser microphone via st.audio_input or file upload) using Whisper.
    """
    model = get_whisper_model()
    if model is None:
        st.error("Whisper package is not installed or model loading failed.")
        return ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        audio_filepath = temp_file.name

    try:
        result = model.transcribe(audio_filepath)
        return result.get("text", "").strip()
    except Exception as e:
        st.error(f"Error during audio transcription: {e}")
        return ""
    finally:
        try:
            os.remove(audio_filepath)
        except Exception:
            pass


def generate_tts(text, lang="en"):
    """
    Generates text-to-speech audio for a response.
    First tries ElevenLabs (if key is available), then falls back to gTTS (100% free, no key needed).
    Returns path to temporary MP3 file, or None.
    """
    if not text or not text.strip():
        return None

    # Try ElevenLabs first if key is available
    if ELEVENLABS_API_KEY:
        try:
            audio_path = elevenlabs_tts(text, lang=lang)
            if audio_path:
                return audio_path
        except Exception as e:
            print(f"ElevenLabs TTS failed: {e}")

    # Fallback to gTTS
    if gTTS is not None:
        try:
            lang_map = {
                "zh-cn": "zh-CN",
                "auto": "en"
            }
            gtts_lang = lang_map.get(lang.lower(), lang[:2].lower())
            tts = gTTS(text=text[:800], lang=gtts_lang, slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            print(f"gTTS error: {e}")

    return None


def extract_text_from_file(uploaded_file):
    """
    Extracts text content from uploaded PDF, TXT, MD, or HTML files.
    """
    file_content = uploaded_file.getvalue()
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "pdf":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            doc = fitz.open(temp_file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            os.unlink(temp_file_path)
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    elif file_extension in ["txt", "md", "html"]:
        return file_content.decode("utf-8")
    else:
        st.warning(f"Unsupported file type: {file_extension}. Please upload PDF, TXT, MD, or HTML.")
        return ""

def get_ai_response(query, lang="en"):
    """
    Generates an AI response to a query using a ConversationalRetrievalChain with Gemini.
    Translates query to English for LLM, and response back to target language.
    """
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return "Error loading knowledge base. Please try again later.", []

    query_en = GoogleTranslator(source=lang, target="en").translate(query) if lang != "en" else query

    try:
        # Initialize Gemini Chat
        # Using gemini-1.0-flash as it was the last tested. You can try "gemini-pro" again
        # or another supported model if needed.
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Combine the prompts correctly using from_messages
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        result = qa_chain({"question": query_en, "chat_history": st.session_state.conversation_history})

        answer_en = result["answer"]
        sources = [doc.page_content[:150] + "..." for doc in result.get("source_documents", [])]

        answer = GoogleTranslator(source="en", target=lang).translate(answer_en) if lang != "en" else answer_en

        st.session_state.conversation_history.append((query_en, answer_en))

        return answer, sources
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}", []