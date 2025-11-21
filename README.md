# Accelerate Contract Analysis with Innovative Turbocharged AI

ContractBot is an enterprise-grade AI platform built with large language models (LLMs) for legal, procurement, and business teams. It transforms the traditionally manual and time-consuming contract review process into an efficient, accurate, and continuously learning workflow.

With ContractBot, users can not only instantly compare versions of contracts, but also automatically identify risks, receive professional revision suggestions, and build an evolving internal legal knowledge base. 

## Key Features

-   **📝 AI-Driven Contract Comparison & Analysis (`pages/1_Analysis.py`)**
    -   **Dynamic Comparison**：Upload your standard contract template and a draft under review; AI performs clause-by-clause semantic comparison.
    -   **Risk Summary Matrix**：Automatically generate a clear table summarizing Key Differences and Core Revision Recommendations.。
    -   **Detailed Itemized Reports**：For each review item (e.g., confidentiality term, liability cap), the system generates a report with Clause Summary, Risk Analysis, and Specific Revision Suggestions.
    -   **Continuous Learning**：High-quality reports can be archived with one click; the system integrates them into the AI’s long-term memory, improving future analyses.

-   **🚨 Contract Risk Classifier (`pages/2_Risk_Classifier.py`)**
    -   **Intelligent Clause Segmentation**：Uses semantic, regex, and other algorithms to split contracts into precise clauses.
    -   **Three-Level Risk Rating**：Based on an internal knowledge base (Risk_Knowledge.py), clauses are automatically flagged as High Risk, Medium Risk, or Needs Attention.
    -   **PDF Highlighting**：Generates annotated PDFs with highlighted clauses, risk reasons, and revision suggestions for easy offline review.

-   **🤖 Contract ChatBot (`pages/3_ContractBot.py`)**
    -   **Integrated Knowledge Retrieval**：Query both the permanent knowledge base (Pinecone) and temporary uploaded files (FAISS) via natural language.
    -   **Multi-Document Q&A**：Ask cross-document questions, e.g., “Compare the liability cap in Contract A and Contract B.”
    -   **Traceability**：Every answer includes references to its source for validation.

-   **📊 Control Center (`pages/4_Control_Center.py`)**
    -   **Audit Trail**：All analysis operations are automatically logged for auditing.
    -   **Pinning & Tagging**：Highlight and organize important results with pins and tags.
    -   **One-Click Export**：Export history or pinned items to JSON or CSV.

-   **⚙️ Knowledge Base Admin Dashboard (`pages/5_Admin_Console.py`)**
    -   **Experience Extraction & Injection**：Upload Word files with tracked changes or comments; the system extracts review expertise and feeds it into the knowledge base.
    -   **Index Maintenance**：Admin tools for managing vector database content safely.

## Project Structure
```
/contractbot/
├── Homepage.py              # Application entry point
├── pages/
│   ├── 1_Compare&Analyze.py       # Contract comparison & analysis
│   ├── 2_Automated_Risk_Finder.py # Automated risk detection
│   ├── 3_Contract_ChatBot.py      # Interactive chatbot
│   ├── 4_History&Export.py        # History logging & export
│   └── 5_System_Settings.py       # Settings & knowledge base mgmt
├── utils.py                  # Shared utilities (Pinecone, DOCX handling)
├── storage_utils.py          # Amazon S3 storage functions
├── Risk_Knowledge.py         # Risk classification rules & knowledge
├── text_splitter.py          # Smart text splitting tool
├── logo.png                  # Application logo
├── README.md                 # Documentation
├── requirements.txt          # Dependencies
├── .env                      # API keys & environment variables
├── NotoSansTC-Var.ttf        # Font for PDF outputs
└── .gitignore

```

## Application Architecture & Technology Stack

### RAG (Retrieval-Augmented Generation) Architecture

ContractBot leverages RAG to ground LLM outputs in factual, document-based evidence, improving reliability, compliance, and transparency.

**Core Benefits of RAG:**

1.  **Grounded in Facts**：Traditional LLMs may sometimes fabricate information (a phenomenon known as “hallucination”). The RAG architecture addresses this by first retrieving relevant passages from a specified knowledge base (e.g., your contract documents) and then providing these passages as context to the LLM. This forces the model to generate answers based on factual evidence, significantly improving the reliability of its responses.
2.  **Leverages Private Data**：Enterprises do not need to spend enormous resources retraining a model. RAG enables existing powerful LLMs to directly access and leverage your private, continuously updated internal documents, turning them into true experts on your business.
3.  **Traceable & Transparent**：Since the answers are based on specific retrieved documents, we can easily trace back to the source of the answer, making it convenient for users to verify the accuracy of the information.


#### Dual-Database RAG (Retrieval-Augmented Generation) Strategy

To perfectly distinguish between the two core needs of “real-time analysis” and “long-term knowledge management,” this project adopts an innovative dual-vector database strategy:

-   **FAISS (Temporary Index)**
    -   **Purpose**：Used for immediate, session-level comparisons.
    -   **Characteristics**：When a user uploads two documents for real-time comparison, the system quickly creates a temporary FAISS index in memory. This provides a fast and isolated analysis environment, ensuring that each comparison is independent and that the index is destroyed after the session ends, thereby protecting data privacy.

-   **Pinecone (Permanent Vector Database)**
    -   **Purpose**：Core of system-level long-term memory.
    -   **Characteristics**：Used to store standard contract templates, GCO review experience, and high-quality analysis examples approved by users. Functions like `2_Contract_Bot.py` will retrieve information from here, enabling the AI to have cross-temporal learning and memory capabilities, becoming smarter with continued use.

### Tech Stack

-   **Frontend Framework**: `Streamlit`
-   **AI/LLM Core**: `LangChain`, `OpenAI (gpt-4o, text-embedding-3-small)`
-   **Databases**:
    -   `FAISS` (temporary, in-memory index)
    -   `Pinecone` (permanent vector database)
    -   `Amazon S3` (Persistent file repository used for archiving analysis reports in Markdown format)
-   **Document Processing**: `PyMuPDF (fitz)`, `python-docx`, `lxml`, `unstructured`
-   **Text Splitting**: `semantic-text-splitter`, `spaCy`, `RecursiveCharacterTextSplitter`
-   **Data Validation**: `pydantic` (Used for output validation of `Risk_Classification`)

## Installation & Deployment

### 1. Prerequisites

-   Verify that Python 3.9+ is installed.
-   Obtain API keys for the following services:
    -   `OPENAI_API_KEY`
    -   `PINECONE_API_KEY`
    -   `AWS_ACCESS_KEY_ID` (Used for report archiving)
    -   `AWS_SECRET_ACCESS_KEY` (Used for report archiving)

### 2. Local Setup

**A. Clone the project**
```bash
git clone [https://github.com/hayamayama/project_contractbot.git](https://github.com/hayamayama/project_contractbot.git)
cd project_contractbot
```

**B. Install dependencies**
```bash
pip install -r requirements.txt
```

**C. Download spaCy models**
This project uses spaCy for more accurate text processing. Please run the following command to download the required model:
```bash
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

**D. Configure environment variables (.env)**
In the project root directory, create a file named .env and add the following content:

```env
# .env

# OpenAI API Key
OPENAI_API_KEY="sk-..."

# Pinecone Configuration
PINECONE_API_KEY="..."
PINECONE_INDEX_NAME="contract-assistant" # The index name you created in Pinecone.

# AWS S3 Configuration (Used for report archiving and AI learning functionality)
# If not required, this field may be left blank, but related functionality will be limited.
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
S3_BUCKET_NAME="..."
S3_REGION_NAME="..." # e.g., "ap-northeast-1"
```

### 3. Start the application

In the project root directory, run the following command:

```bash
streamlit run Homepage.py
```

The application will open in your local browser (most likely `http://localhost:8501`).

## Usage flow overview

1.  **[First Use] Establish the knowledge base baseline**
    -   Navigate to **Admin Console (System_Settings)** and upload your company’s standard contract templates or previously reviewed cases (.docx) to establish the AI’s retrieval knowledge base.
    -   You can also upload reference documents on the **AI Contract Pre-Review (Compare&. Analyze)** page to serve as a basis for real-time comparison.

2.  **【Daily Usage】Perform Contract Comparison**
    -   Navigate to **AI Contract Pre-Review (Compare&Analyze)** page.
    -   **Step 1 & 2**：Upload and select a reference document as the comparison baseline.
    -   **Step 3**：(Optional) Adjust the AI analysis Temperature and Max Tokens.
    -   **Step 4**：Upload the document for review and click Start AI In-Depth Review.
    -   After a brief wait, the system will generate a complete risk summary and an itemized analysis report.

3.  **【Advanced Application】Chat with Your Data/Document**
    -   Navigate to **Contract_ChatBot**.
    -   In the sidebar, select the knowledge base(s) you want to query (multiple selections allowed) or temporarily upload a PDF.
    -   In the dialogue box, ask questions directly in natural language, for example: “How long does the confidentiality obligation last after contract termination?”

4.  **【Quick Scan】Automatic Risk Classification.**
    -   Navigate to **Automated_Risk_Finder**.
    -   Upload a contract PDF and click Analyze.
    -   The system will automatically list all high- and medium-risk clauses and provide a downloadable PDF file with key highlights.

## Streamlit Community Cloud Link
[https://project-contractbot.streamlit.app/](https://projectcontractbotai-6pwpf2ag5ngaefdjoaqsdf.streamlit.app/)

## Licensing

©2025 Ernst & Young LLP. All Rights Reserved.
