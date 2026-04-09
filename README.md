🏥 Dual AI Chat Assistant (Medical RAG + General AI)

A powerful Streamlit-based AI application that combines:

- 🧠 Medical RAG (Retrieval-Augmented Generation) for pharmacy-related queries
- 🤖 General AI Assistant for open-ended questions

Built using LangChain, Groq LLM, ChromaDB, and Ollama Embeddings, this project delivers a real-time, WhatsApp-style chat experience.

---

🚀 Features

🏥 Medical RAG Assistant

- Answers medicine-related queries using a local vector database
- Performs:
  - ✅ Stock availability check
  - 💊 Dosage instructions
  - 🔄 Alternative medicine suggestion
  - 📚 FAQ-style responses
- Ensures context-based accurate answers

---

🤖 General AI Assistant

- Handles any general query
- Powered by Groq LLaMA 3.1
- Fast and conversational responses

---

💬 UI Highlights

- WhatsApp-style chat interface
- Dual chat window (side-by-side assistants)
- Smooth real-time interaction

---

🛠️ Tech Stack

- Frontend/UI: Streamlit
- LLM: Groq (LLaMA 3.1 8B Instant)
- Embeddings: Ollama ("nomic-embed-text")
- Vector DB: ChromaDB
- Framework: LangChain

---

📂 Project Structure

RAG/
│── app.py                  # Main Streamlit app
│── ingest_data.py         # Script to create vector DB
│── requirements.txt       # Dependencies
│── chroma_db_medical_5k/  # Vector database
│── customized_dataset.csv # Medical dataset
│── README.md              # Project documentation

---

⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-username/dual-ai-chat-assistant.git
cd dual-ai-chat-assistant

---

2️⃣ Create Virtual Environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

---

3️⃣ Install Dependencies

pip install -r requirements.txt

---

4️⃣ Setup Ollama (for Embeddings)

Install Ollama from: https://ollama.com

Then run:

ollama pull nomic-embed-text

---

5️⃣ Add Groq API Key

Replace this line in "app.py":

os.environ["GROQ_API_KEY"] = "your_api_key_here"

Get API key from: https://console.groq.com

---

6️⃣ Create Vector Database

Run:

python ingest_data.py

This will generate:

chroma_db_medical_5k/

---

7️⃣ Run the App

streamlit run app.py

---

📊 How It Works

🔹 Medical RAG Pipeline

1. User asks a query
2. Query is converted into embeddings
3. ChromaDB retrieves relevant documents
4. Context + Query → Groq LLM
5. AI generates accurate response

---

🔹 General AI Pipeline

1. User query → Prompt template
2. Directly sent to Groq LLM
3. Returns fast conversational answer

---

🧪 Example Queries

Medical Assistant

- "Do you have Paracetamol in stock?"
- "Medicine for fever"
- "Dosage of Amoxicillin"
- "Alternative for Ibuprofen"

General Assistant

- "Explain machine learning"
- "What is Python used for?"
- "Tell me a joke"

---

⚠️ Important Notes

- ❗ Do NOT upload your API key to GitHub
- Use ".env" file for production
- Ensure "chroma_db_medical_5k" exists before running

---

🔐 Security Best Practice

Instead of hardcoding API key:

pip install python-dotenv

Create ".env":

GROQ_API_KEY=your_key_here

Then in code:

from dotenv import load_dotenv
load_dotenv()

---

📈 Future Improvements

- ✅ Add voice input
- 📊 Add analytics dashboard
- 🔍 Improve retrieval with hybrid search
- 📱 Mobile-friendly UI
- 🧠 Multi-dataset RAG

---

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first.

---

📄 License

This project is licensed under the MIT License.

---

👨‍💻 Author

Bibek
AI/ML Developer | Full Stack Enthusiast

---

⭐ Support

If you like this project:

- ⭐ Star the repo
- 🍴 Fork it
- 📢 Share it

---

🔥 Built for real-world AI + healthcare applications