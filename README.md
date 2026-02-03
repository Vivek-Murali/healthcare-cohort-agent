# Clinical Graph Intelligence: E2E Patient Cohort Discovery & Risk Prediction

An end-to-end Healthcare Intelligence platform that transforms unstructured clinical notes into a Knowledge Graph (KG) to perform cohort analysis, Graph Neural Network (GNN) based risk prediction, and Agentic RAG-driven insights.

## 🚀 Overview
This project extends the Databricks/John Snow Labs (JSL) framework into a production-grade architecture. It bridges the gap between raw medical text and actionable clinical decisions using a "Lakehouse + Graph" approach.

### Key Features
* **NLP Extraction:** Automated NER and Relation Extraction from clinical notes using **Spark NLP**.
* **Knowledge Graph:** Multi-modal graph storage in **Neo4j** (Patients, Diseases, Medications, Procedures).
* **GNN Risk Modeling:** A **GraphSAGE** model built with **PyTorch Geometric** to predict future patient-disease comorbidities.
* **Agentic Interface:** A **LangChain-powered Agent** that uses "Thought-Action-Observation" loops to query the graph and provide clinical summaries via LLMs.

---

## 🏗️ Architecture
1. **Silver Layer (Databricks):** Processed Delta Tables + Spark NLP for entity extraction.
2. **Gold Layer (Neo4j):** Graph projection of patient-centric relationships.
3. **ML Ops (MLflow):** GNN model versioning and hyperparameter tracking.
4. **Agent Layer (Streamlit/Groq):** A conversational interface for clinicians to ask complex cohort questions.

---

## 🛠️ Tech Stack
* **Data Orchestration:** Databricks, Spark NLP, Delta Lake
* **Graph Database:** Neo4j AuraDB
* **Machine Learning:** PyTorch Geometric (GNN), MLflow
* **AI Agents:** LangChain, Llama 3 (via Groq), Streamlit
* **Language:** Python, Cypher (Graph Query)

---

## 📊 GNN Implementation
The GNN treats cohort discovery as a **Link Prediction** task. By analyzing the neighborhood of a Patient node, the model predicts the probability of a `DIAGNOSED_WITH` relationship for conditions not yet present in the patient's history.

$$P(e_{u,v}) = \sigma(z_u^T z_v)$$

*Where $z_u$ and $z_v$ are embeddings generated via Graph Attention layers.*

---

## 🤖 Agent Capabilities
The Agent is equipped with three tools:
1.  **CypherTool:** Converts natural language to Graph queries.
2.  **RiskPredictor:** Fetches inference results from the GNN model.
3.  **MedicalSummarizer:** Summarizes findings using a HIPAA-aware prompting strategy.

---

## 📂 Project Structure
```text
├── notebooks/
│   ├── 01_etl_extraction.ipynb    # Spark NLP & Delta Lake
│   ├── 02_graph_construction.ipynb # Neo4j Sink
│   └── 03_gnn_training.py         # PyTorch Geometric Model
├── agent/
│   ├── tools.py                   # Custom LangChain tools
│   └── app.py                     # Streamlit UI
└── README.md
```

## 🏁 Getting Started
1. Clone the repo.
2. Import the notebooks/ into your Databricks Community Edition.
3. Set up a free Neo4j AuraDB instance and add credentials to your environment variables.
4. Run streamlit run agent/app.py.

