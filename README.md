Maintenance Computer Vision Assistant
A Streamlit application that uses computer vision and AI to automatically identify maintenance issues and classify them into appropriate work types.


Features

Video Analysis: Upload videos of maintenance issues for automatic analysis
Computer Vision: Frame extraction and analysis using Groq's vision AI
Issue Detection: Automatic highlighting of potential maintenance problems in frames
Work Type Classification: AI-powered matching to the appropriate maintenance work type
Detailed Reporting: Comprehensive analysis with confidence scores and reasoning

Demo


Architecture
This application combines several technologies:

Streamlit for the web interface
OpenAI Embeddings for semantic search capabilities
Groq LLM and Vision API for natural language understanding and visual analysis
Pinecone Vector Database for embedding storage and similarity search
OpenCV for video processing and frame analysis

Getting Started
Prerequisites

Python 3.8+
Pinecone account
OpenAI API key
Groq API key

Installation

Clone the repository:

bashgit clone https://github.com/yourusername/maintenance-vision-assistant.git
cd maintenance-vision-assistant

Create a virtual environment and install dependencies:

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Create a .env file in the project root with your API keys:

PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key

Place your work types CSV data in the project root directory as AllWorktypesAndDetails.csv
Add a logo image named ai_logo.png in the project root for the custom assistant avatar

Running the Application
bashstreamlit run app.py
How It Works

Video Upload: The user uploads a video of a maintenance issue
Frame Extraction: The system extracts key frames from the video
Visual Analysis: Each frame is analyzed using computer vision
Issue Detection: Potential maintenance issues are highlighted
Work Type Matching: The system matches the visual data to potential work types
Final Classification: The most appropriate work type is identified with a confidence score

Usage Examples
Video Analysis
Upload a video showing a maintenance issue such as:

Water leaks
Electrical problems
Structural damage
HVAC issues

The system will:

Extract and display key frames
Analyze each frame
Highlight potential issues
Generate a summary of the problem
Identify the appropriate work type

Project Structure
maintenance-vision-assistant/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Project dependencies
├── AllWorktypesAndDetails.csv  # Work type data
├── ai_logo.png              # Custom avatar image
├── .env                     # Environment variables (not tracked by git)
├── .gitignore               # Specifies intentionally untracked files
├── assets/                  # Images for README and documentation
└── README.md                # Project documentation
Future Improvements

Real-time video analysis
Multi-issue detection in a single video
Mobile compatibility for field technicians
Integration with work order systems
Extended work type database

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Streamlit for the web application framework
OpenAI for the embedding model
Groq for the LLM and vision models
Pinecone for the vector database
OpenCV for computer vision capabilities