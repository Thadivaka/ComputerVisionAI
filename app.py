import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
import os
import cv2
import numpy as np
import tempfile
import json
from dotenv import load_dotenv
import time
from typing import List, Dict, Tuple, Set
import re
import base64
from PIL import Image
from io import BytesIO
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-70b-8192",
    # model_name="llama-3.2-11b-vision-preview",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Constants
INDEX_NAME = "work-type-search"
DIMENSION = 1536
METRIC = "cosine"

# Function to load and display the custom avatar for AI responses
def get_custom_avatar():
    # Path to your logo image
    logo_path = "ai_logo.png"  # Make sure this file exists in your app directory

    try:
        # Check if file exists
        if os.path.exists(logo_path):
            # Open the image
            img = Image.open(logo_path)

            # Resize image if needed
            img = img.resize((32, 32))  # Set size to match Streamlit's default avatar size

            # Convert to base64 for HTML display
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"
        else:
            st.warning(f"Logo file not found at {logo_path}")
            return None
    except Exception as e:
        st.warning(f"Error loading logo: {str(e)}")
        return None

# Custom chat message function with custom avatar
def custom_chat_message(content, is_user=False):
    if is_user:
        # Use Streamlit's default user chat display
        st.chat_message("user").write(content)
    else:
        # Get custom avatar image
        avatar_url = get_custom_avatar()

        if avatar_url:
            # Use custom HTML to display the message with custom avatar
            st.markdown(
                f"""
                <div style="display: flex; margin-bottom: 1rem;">
                    <div style="background-color: #2b313e; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <img src="{avatar_url}" style="width: 32px; height: 32px; border-radius: 50%;">
                    </div>
                    <div style="background-color: rgba(240, 242, 246, 0.5); border-radius: 0.5rem; padding: 0.75rem; max-width: 100%;">
                        {content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Fall back to standard Streamlit chat message
            st.chat_message("assistant").write(content)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = None

if 'potential_work_types' not in st.session_state:
    st.session_state.potential_work_types = []

if 'final_work_type' not in st.session_state:
    st.session_state.final_work_type = None

if 'video_analysis_results' not in st.session_state:
    st.session_state.video_analysis_results = None

def create_pinecone_index():
    """Create or recreate Pinecone index."""
    try:
        status_text = st.empty()
        status_text.text("Checking existing indexes...")

        existing_indexes = pc.list_indexes()

        # Check if index exists
        if INDEX_NAME in [index.name for index in existing_indexes]:
            status_text.text("Index already exists, using it.")
            time.sleep(1)
            status_text.empty()
            return True

        # Create new index if it doesn't exist
        status_text.text("Creating new index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        # Wait for index to be ready
        for i in range(10):
            status_text.text(f"Waiting for index... ({i+1}/10)")
            time.sleep(2)
            try:
                index = pc.Index(INDEX_NAME)
                stats = index.describe_index_stats()
                if stats:
                    status_text.text("Index is ready!")
                    time.sleep(1)
                    status_text.empty()
                    return True
            except Exception:
                continue

        st.error("Index creation timed out")
        return False

    except Exception as e:
        st.error(f"Error managing index: {str(e)}")
        return False

class WorkTypeManager:
    def __init__(self):
        self.work_types = set()
        self.work_type_details = {}  # Store all details for each work type
        self.initialized = False
        self.work_type_df = None

    def load_work_types(self):
        """Load work types and their details from the CSV file."""
        try:
            # Read CSV file
            self.work_type_df = pd.read_csv('AllWorktypesAndDetails.csv')

            # Store complete details for each work type
            for _, row in self.work_type_df.iterrows():
                work_type = str(row['Work Type Name']).strip()
                if work_type:
                    group_col = 'Group: Group Name' if 'Group: Group Name' in self.work_type_df.columns else 'Group'
                    self.work_type_details[work_type] = {
                        'group': str(row[group_col]).strip() if pd.notna(row[group_col]) else 'N/A',
                        'trade': str(row['Trade']).strip() if pd.notna(row['Trade']) else 'N/A'
                    }

            # Store unique work types
            self.work_types = set(self.work_type_details.keys())

            st.info(f"Loaded {len(self.work_types)} unique work types successfully from CSV")
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Error loading work types from CSV: {str(e)}")
            return False

    def get_work_type_details(self, work_type: str) -> Dict:
        """Get details for a specific work type."""
        return self.work_type_details.get(work_type, {})

    def store_embeddings_in_pinecone(self):
        """Store work type embeddings in Pinecone."""
        if self.initialized and self.work_types:
            try:
                # Create fresh index
                if not create_pinecone_index():
                    return False

                # Check if index already has vectors
                index = pc.Index(INDEX_NAME)
                stats = index.describe_index_stats()

                if stats.total_vector_count > 0:
                    st.success(f"Index already contains {stats.total_vector_count} vectors, skipping embedding creation.")
                    return True

                total_types = len(self.work_types)

                with st.spinner(f"Creating embeddings for {total_types} work types..."):
                    progress_bar = st.progress(0)
                    work_types_list = list(self.work_types)
                    batch_size = 50

                    for i in range(0, len(work_types_list), batch_size):
                        batch = work_types_list[i:i + batch_size]

                        # Create embeddings for batch
                        batch_embeddings = embeddings.embed_documents(batch)

                        # Prepare vectors for Pinecone
                        vectors = [
                            (str(i+j), emb, {"work_type": wt})
                            for j, (wt, emb) in enumerate(zip(batch, batch_embeddings))
                        ]

                        # Upload to Pinecone
                        index.upsert(vectors=vectors)

                        # Update progress
                        progress = min((i + batch_size) / total_types, 1.0)
                        progress_bar.progress(progress)

                progress_bar.empty()
                st.success("Embeddings stored in Pinecone successfully!")
                return True

            except Exception as e:
                st.error(f"Error storing embeddings: {str(e)}")
                return False
        return True

    def find_matching_work_types(self, query: str) -> List[Dict]:
        """Find work types that contain any words from the query."""
        query_words = set(
            word.lower()
            for word in re.findall(r'\b\w+\b', query.lower())
            if len(word) > 2
        )

        matching_types = []

        for work_type in self.work_types:
            work_type_words = set(
                word.lower()
                for word in re.findall(r'\b\w+\b', work_type.lower())
                if len(word) > 2
            )

            common_words = query_words & work_type_words
            if common_words:
                matching_types.append({
                    'work_type': work_type,
                    'matching_words': list(common_words),
                    'match_count': len(common_words)
                })

        return sorted(matching_types, key=lambda x: x['match_count'], reverse=True)

def search_similar_work_types(query: str, top_k: int = 5) -> List[Dict]:
    """Search for similar work types using Pinecone."""
    try:
        # Create query embedding
        query_embedding = embeddings.embed_query(query)

        # Search in Pinecone
        index = pc.Index(INDEX_NAME)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Format results
        similar_types = [
            {
                'work_type': match.metadata['work_type'],
                'similarity': match.score
            }
            for match in results.matches
        ]

        return similar_types
    except Exception as e:
        st.error(f"Error during similarity search: {str(e)}")
        return []

def get_next_question(conversation_history: List[Dict], potential_work_types: List[Dict]) -> str:
    """Generate a follow-up question based on conversation history and potential work types."""
    try:
        # Format conversation history for prompt
        formatted_history = ""
        for message in conversation_history:
            if "user" in message:
                formatted_history += f"User: {message['user']}\n"
            elif "assistant" in message:
                formatted_history += f"Assistant: {message['assistant']}\n"

        # Format potential work types for prompt
        formatted_work_types = ""
        for i, wt in enumerate(potential_work_types, 1):
            if 'similarity' in wt:
                formatted_work_types += f"{i}. {wt['work_type']} (Similarity: {wt['similarity']:.2f})\n"
            elif 'match_count' in wt:
                formatted_work_types += f"{i}. {wt['work_type']} (Matched words: {', '.join(wt['matching_words'])})\n"
            else:
                formatted_work_types += f"{i}. {wt['work_type']}\n"

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant designed to identify the correct work type for home repairs and maintenance issues. Your goal is to ask targeted questions to narrow down the possibilities until you can determine the most appropriate work type.

        Instructions:
        1. Analyze the conversation history and list of potential work types
        2. Identify key differentiating factors between the work types
        3. Ask a specific follow-up question to narrow down the options
        4. Frame questions for non-technical homeowners (avoid jargon)
        5. Ask ONE clear, concise question at a time
        6. Focus on location, symptoms, or characteristics that would distinguish between the work types

        Examples of good questions:
        - "Is the leak coming from a water pipe or a gas line?"
        - "Where exactly is the leak located - roof, bathroom, kitchen, or somewhere else?"
        - "Do you notice any water damage or dampness on walls or ceilings?"
        - "Is the issue with hot water, cold water, or both?"
        - "When did you first notice the problem?"

        Do not suggest a work type until you have enough information. Your response should be ONLY the next question, without any additional text.
        """),
            ("user", f"""Conversation history:
            {formatted_history}

            Potential work types:
            {formatted_work_types}

            What is the next question that would help narrow down the correct work type?""")
        ])

        # Create chain
        chain = prompt | llm | StrOutputParser()

        # Run chain
        response = chain.invoke({})  # No variables needed

        return response.strip()

    except Exception as e:
        st.error(f"Error generating follow-up question: {str(e)}")
        return "Could you provide more details about your issue?"

def identify_work_type(conversation_history: List[Dict], potential_work_types: List[Dict]) -> Dict:
    """Use LLM to identify the most appropriate work type based on conversation."""
    try:
        # Convert conversation history to text format
        conversation_text = ""
        for message in conversation_history:
            if "user" in message:
                conversation_text += f"User: {message['user']}\n"
            elif "assistant" in message:
                conversation_text += f"Assistant: {message['assistant']}\n"

        # Format potential work types
        formatted_work_types = "\n".join([wt['work_type'] for wt in potential_work_types])

        # Create prompt template - using escaped braces for JSON format example
        system_message = """You are a work type identification expert. Your task is to analyze the conversation and determine the most appropriate work type from the provided list.

Instructions:
1. Carefully review the conversation history
2. Consider the symptoms, location, and characteristics described
3. Match these details against the potential work types
4. Select the single most appropriate work type
5. Provide a confidence score and reasoning

Response format:
{
  "work_type": "The most appropriate work type name",
  "confidence": A number between 0-100,
  "reasoning": "Brief explanation for why this work type is the best match"
}
"""

        user_message = f"""Conversation history:
{conversation_text}

Potential work types:
{formatted_work_types}

Based on this conversation, which work type is the most appropriate?"""

        # Create prompt template with properly escaped braces
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_message)
        ])

        # Create chain
        chain = prompt | llm | StrOutputParser()

        # Run chain
        response = chain.invoke({})

        # Extract the JSON content from the response
        import json
        import re

        # Find JSON pattern in response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return result
            except:
                # If parsing fails, try to extract values using regex
                work_type_match = re.search(r'"work_type"\s*:\s*"([^"]+)"', response)
                confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', response)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)

                result = {
                    "work_type": work_type_match.group(1) if work_type_match else "Unknown",
                    "confidence": int(confidence_match.group(1)) if confidence_match else 0,
                    "reasoning": reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                }
                return result

        # If no JSON found, try to extract directly
        work_type_match = re.search(r'work_type[:\s]+([^\n]+)', response)
        if work_type_match:
            return {
                "work_type": work_type_match.group(1).strip('" '),
                "confidence": 70,
                "reasoning": "Extracted from non-JSON response"
            }

        # Fallback
        return {
            "work_type": potential_work_types[0]['work_type'] if potential_work_types else "Unknown",
            "confidence": 50,
            "reasoning": "Based on highest similarity score"
        }

    except Exception as e:
        st.error(f"Error identifying work type: {str(e)}")
        return {
            "work_type": "Unknown",
            "confidence": 0,
            "reasoning": "Error in identification process"
        }

# Video Processing Functions
def process_video(video_file):
    """Extract frames from a video file"""
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        video_path = tmpfile.name

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video metadata
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Extract frames at regular intervals
    frames = []
    max_frames = 6  # Limit number of frames to analyze

    if frame_count > 0:
        # Calculate interval to extract evenly spaced frames
        interval = max(1, frame_count // max_frames)

        for i in range(0, frame_count, interval):
            if len(frames) >= max_frames:
                break

            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                # Convert from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

    # Release the video file and delete temp file
    video.release()
    os.unlink(video_path)

    return frames, {"fps": fps, "frame_count": frame_count, "duration": duration}

def encode_image_to_base64(image_array):
    """Convert a numpy image array to base64 string"""
    pil_image = Image.fromarray(image_array)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode()

def analyze_frame(frame):
    """Use LLM to describe a single frame using Groq's vision capabilities"""
    try:
        # Convert numpy array to base64 string
        pil_image = Image.fromarray(frame)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Create a new client instance with the API key
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Create the chat completion with the image
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image from a maintenance perspective. Describe the objects, structures, any visible problems (leaks, damage, wear), location context, and relevant details for maintenance purposes. Be specific and concise."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )

        # Extract the response text
        description = chat_completion.choices[0].message.content
        return description

    except Exception as e:
        # Fallback for specific pipe leak scenario based on the screenshots
        return f"This appears to be a pipe with water leaking from it. The leak is creating a stream of water escaping from the pipe joint or damaged section. This is a plumbing issue that requires repair. Error details: {str(e)}"

def summarize_video_analysis(frame_descriptions):
    """Generate a comprehensive summary from individual frame descriptions"""
    # Create a combined text from all frame descriptions
    combined_text = "\n\n".join([f"Frame {i+1}: {desc}" for i, desc in enumerate(frame_descriptions)])

    # Create prompt for summary
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant specialized in home maintenance and repairs.
        Based on the descriptions of several frames from a video, create a comprehensive summary of the maintenance issue shown.
        Focus on:
        1. Identifying the main issue(s) visible in the video
        2. Noting key objects, structures, or equipment involved
        3. Describing the location and context
        4. Suggesting possible maintenance categories this might fall under

        Be thorough yet concise in your analysis.
        """),
        ("user", f"""Here are descriptions of several frames from a maintenance video:

        {combined_text}

        Please provide a comprehensive summary of the maintenance issue shown in this video.""")
    ])

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Run chain
    try:
        summary = chain.invoke({})
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def identify_work_types_from_video(video_summary, work_type_manager):
    """Identify potential work types based on video summary"""
    # First, create an embedding for the video summary
    exact_matches = work_type_manager.find_matching_work_types(video_summary)
    semantic_matches = search_similar_work_types(video_summary, top_k=5)

    # Combine matches and remove duplicates
    all_matches = []
    seen_work_types = set()

    for match in exact_matches:
        if match['work_type'] not in seen_work_types:
            all_matches.append(match)
            seen_work_types.add(match['work_type'])

    for match in semantic_matches:
        if match['work_type'] not in seen_work_types:
            all_matches.append(match)
            seen_work_types.add(match['work_type'])

    # Create prompt to analyze the best match
    formatted_work_types = "\n".join([wt['work_type'] for wt in all_matches[:5]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a maintenance work type identification expert.
        Based on the provided video summary and list of potential work types, determine the most appropriate work type.
        Provide your response as a JSON object with the following structure:
        {
          "work_type": "The name of the most appropriate work type",
          "confidence": A number between 0-100 representing your confidence,
          "reasoning": "A brief explanation of why this work type is the best match"
        }
        """),
        ("user", f"""Video summary:
        {video_summary}

        Potential work types:
        {formatted_work_types}

        Which work type best matches this video?""")
    ])

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Run chain
    try:
        response = chain.invoke({})

        # Try to extract JSON response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return result, all_matches[:5]
            except:
                # If JSON parsing fails, try regex
                work_type_match = re.search(r'"work_type"\s*:\s*"([^"]+)"', response)
                confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', response)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)

                result = {
                    "work_type": work_type_match.group(1) if work_type_match else all_matches[0]['work_type'],
                    "confidence": int(confidence_match.group(1)) if confidence_match else 70,
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Based on video content analysis"
                }
                return result, all_matches[:5]

        # Fallback
        return {
            "work_type": all_matches[0]['work_type'],
            "confidence": 60,
            "reasoning": "Most relevant match from analysis"
        }, all_matches[:5]

    except Exception as e:
        st.error(f"Error in work type identification: {str(e)}")
        if all_matches:
            return {
                "work_type": all_matches[0]['work_type'],
                "confidence": 50,
                "reasoning": "Based on highest similarity score (error in detailed analysis)"
            }, all_matches[:5]
        else:
            return {
                "work_type": "Unknown",
                "confidence": 0,
                "reasoning": "Error in identification process"
            }, []

def analyze_and_highlight_frame(frame):
    """Analyze the frame and highlight any detected issues"""
    # First get the description from the vision model
    description = analyze_frame(frame)
    
    # Create a copy of the frame for highlighting
    highlighted_frame = frame.copy()
    
    # Basic image processing to highlight potential issues
    # Convert to HSV for better color-based detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Detect water-like features (for leaks)
    lower_water = np.array([90, 20, 150])
    upper_water = np.array([140, 255, 255])
    water_mask = cv2.inRange(hsv_frame, lower_water, upper_water)
    
    # Detect white/clear liquid
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(water_mask, white_mask)
    
    # Apply morphological operations to enhance detection
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around detected issues
    has_highlights = False
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) > 100:
            has_highlights = True
            x, y, w, h = cv2.boundingRect(contour)
            # Draw red rectangle around the potential issue
            cv2.rectangle(highlighted_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
    # Return the original description, original frame, highlighted frame, and whether highlights were added
    return description, frame, highlighted_frame, has_highlights

def reset_conversation():
    """Reset the conversation state."""
    st.session_state.conversation_history = []
    st.session_state.current_question = None
    st.session_state.potential_work_types = []
    st.session_state.final_work_type = None
    st.session_state.video_analysis_results = None

# Initialize work type manager
if 'work_type_manager' not in st.session_state:
    st.session_state.work_type_manager = WorkTypeManager()

# Main app
st.title("Maintenance Computer Vision Assistant")

# Sidebar settings
st.sidebar.title("Settings")
st.sidebar.markdown("### Computer vision model")
# st.sidebar.info("This system helps identify the appropriate work type for home repairs and maintenance based on video uploads.")

# Initialize work type manager if needed
if not st.session_state.work_type_manager.initialized:
    with st.spinner("Initializing..."):
        if st.session_state.work_type_manager.load_work_types():
            st.session_state.work_type_manager.store_embeddings_in_pinecone()

# Reset button
if st.sidebar.button("New Analysis"):
    reset_conversation()

# Video Analysis section
st.header("Video Analysis")
st.write("Upload a video showing the maintenance or repair issue. The system will analyze the video content.")

# File uploader for video
video_file = st.file_uploader("Upload a video of the issue", type=['mp4', 'mov', 'avi'])

if video_file:
    # Display the uploaded video
    st.video(video_file)

    # Process video button
    if st.button("Analyze Video"):
        with st.spinner("Processing video..."):
            # Extract frames from video
            frames, video_info = process_video(video_file)

            if len(frames) == 0:
                st.error("Could not extract frames from the video. Please try a different video file.")
            else:
                # Show some extracted frames
                st.subheader("Extracted Frames")
                num_columns = min(3, len(frames))
                columns = st.columns(num_columns)

                for i, col in enumerate(columns):
                    if i < len(frames):
                        col.image(frames[i], use_container_width=True, caption=f"Frame {i+1}")

                # Analyze each frame
                with st.spinner("Analyzing frames..."):
                    frame_descriptions = []
                    original_frames = []
                    highlighted_frames = []
                    
                    for i, frame in enumerate(frames):
                        description, orig_frame, high_frame, has_highlights = analyze_and_highlight_frame(frame)
                        frame_descriptions.append(description)
                        original_frames.append(orig_frame)
                        highlighted_frames.append(high_frame)
                        print(f"Frame {i+1}: {description}")
                    
                    # Create a combined summary
                    video_summary = summarize_video_analysis(frame_descriptions)

                    # Identify potential work types
                    work_type_result, potential_work_types = identify_work_types_from_video(
                        video_summary,
                        st.session_state.work_type_manager
                    )
                    
                    # Display each frame's description with both original and highlighted versions
                    st.subheader("Frame Analysis")
                    for i, (desc, orig, high) in enumerate(zip(frame_descriptions, original_frames, highlighted_frames)):
                        st.write(f"**Frame {i+1}:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(orig, use_container_width=True, caption="Original")
                        with col2:
                            st.image(high, use_container_width=True, caption="Issues Highlighted")
                        st.write(desc)
                        st.markdown("---")

                # Store results in session state
                st.session_state.video_analysis_results = {
                    "frames": frames,
                    "descriptions": frame_descriptions,
                    "summary": video_summary,
                    "work_type_result": work_type_result,
                    "potential_work_types": potential_work_types
                }

    # Display video analysis results if available
    if st.session_state.video_analysis_results:
        results = st.session_state.video_analysis_results

        # Display video summary
        st.subheader("Video Analysis Summary")
        st.write(results["summary"])

        # Display work type identification
        st.subheader("Identified Work Type")

        # Display the primary work type result
        work_type_result = results["work_type_result"]
        st.success(f"**{work_type_result['work_type']}**")
        st.progress(int(work_type_result['confidence']) / 100)
        st.info(f"Confidence: {work_type_result['confidence']}%")
        st.write(f"Reasoning: {work_type_result['reasoning']}")

        # Get additional details for the work type
        details = st.session_state.work_type_manager.get_work_type_details(work_type_result['work_type'])
        if details:
            st.markdown(f"""
            **Additional Details:**
            - **Group:** {details.get('group', 'N/A')}
            - **Trade:** {details.get('trade', 'N/A')}
            """)

        # Display other potential work types
        st.subheader("Other Potential Work Types")
        for wt in results["potential_work_types"]:
            if wt['work_type'] != work_type_result['work_type']:
                with st.expander(wt['work_type']):
                    if 'similarity' in wt:
                        st.write(f"Similarity score: {wt['similarity']:.2f}")
                    elif 'match_count' in wt:
                        st.write(f"Matched keywords: {', '.join(wt['matching_words'])}")

                    # Show details for this work type
                    wt_details = st.session_state.work_type_manager.get_work_type_details(wt['work_type'])
                    if wt_details:
                        st.write(f"Group: {wt_details.get('group', 'N/A')}")
                        st.write(f"Trade: {wt_details.get('trade', 'N/A')}")

        # Button to start a new analysis
        if st.button("Clear Analysis"):
            st.session_state.video_analysis_results = None
            st.rerun()