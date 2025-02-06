# Import necessary libraries
from transformers import pipeline
import pandas as pd
from neo4j import GraphDatabase
import gradio as gr
import re
import json

# Configuration for Neo4j connection and CSV file path
NEO4J_URI = 'neo4j+s://9094f4bc.databases.neo4j.io'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'djWulnOU1qIkc5j_64OUnlI7h6EJjoOQjJ5O_0IbbmE'
CSV_PATH = "detection_logs.csv"

class Neo4jExpert:
    def __init__(self):
        # Initialize the Neo4j driver for database connection
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Initialize the text generation pipeline with an open-source model
        self.cypher_pipe = pipeline(
            "text2text-generation",
            model="t5-base",  # Open-source model suitable for text generation tasks
            max_length=100
        )
        
    def fetch_all_nodes(self):
        """
        Fetches all nodes from the Neo4j database to understand the data structure.
        """
        query = "MATCH (n) RETURN n LIMIT 10"  # Limit to 10 for brevity
        with self.driver.session() as session:
            result = session.run(query)
            nodes = [record["n"] for record in result]
        return nodes

    def generate_cypher(self, question):
        """
        Generates a Cypher query based on the input question using the text generation model.
        """
        prompt = f"""
        Translate the following natural language question into a Cypher query:
        
        Question: {question}
        Cypher:
        """
        # Generate the Cypher query using the model
        generated = self.cypher_pipe(prompt)[0]['generated_text']
        return generated.strip()

    def validate_query(self, query):
        """
        Validates the generated Cypher query to ensure it is safe and starts with 'MATCH'.
        """
        if not re.match(r"^MATCH\s", query, re.IGNORECASE):
            return False, "Query must start with MATCH"
        if any(kw in query.upper() for kw in ["CREATE", "DELETE", "DROP"]):
            return False, "Write operations are not allowed"
        return True, ""

    def execute_query(self, query):
        """
        Executes the validated Cypher query against the Neo4j database.
        """
        try:
            valid, msg = self.validate_query(query)
            if not valid:
                return msg
            
            with self.driver.session() as session:
                result = session.run(query)
                return [dict(record) for record in result]
        except Exception as e:
            return f"Execution Error: {str(e)}"

class CSVExpert:
    def __init__(self):
        # Initialize the question-answering pipeline with an open-source model
        self.qa_pipe = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"  # Open-source model for QA tasks
        )

    def answer_question(self, question):
        """
        Answers a question based on the content of the CSV file.
        """
        try:
            # Load the CSV data into a DataFrame
            df = pd.read_csv(CSV_PATH)
            # Combine relevant columns to create a context for the QA model
            context = "\n".join([
                f"Detection {idx+1}: {row['Person ID']} at {row['Detection Time']}"
                for idx, row in df.iterrows()
            ])
            # Use the QA model to find the answer within the context
            return self.qa_pipe(question=question, context=context)["answer"]
        except Exception as e:
            return f"CSV Error: {str(e)}"

class ChatBot:
    def __init__(self):
        # Initialize the Neo4j and CSV experts
        self.neo4j = Neo4jExpert()
        self.csv = CSVExpert()

    def handle_query(self, message, source):
        """
        Handles user queries by directing them to the appropriate data source (Neo4j or CSV).
        """
        if source == "Neo4j":
            # Fetch all nodes to understand the data structure
            nodes = self.neo4j.fetch_all_nodes()
            # Generate a Cypher query based on the user's question
            cypher = self.neo4j.generate_cypher(message)
            # Execute the generated Cypher query
            result = self.neo4j.execute_query(cypher)
            
            if isinstance(result, list):
                return f"🔷 Neo4j Result:\n{json.dumps(result[:3], indent=2)}\n\nGenerated Cypher:\n{cypher}"
            return f"❌ Neo4j Error: {result}"
        
        elif source == "CSV":
            # Answer the question based on the CSV data
            return f"📊 CSV Answer: {self.csv.answer_question(message)}"
        
        return "Please select a valid data source."

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔍 Data Query Assistant")
    
    with gr.Row():
        data_source = gr.Radio(
            ["Neo4j", "CSV"],
            label="Data Source",
            value="Neo4j"
        )
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Question")
    clear = gr.Button("Clear")
    
    def respond(message, history, source):
        bot = ChatBot()
        response = bot.handle_query(message, source)
        return response

    msg.submit(
        respond, 
        [msg, chatbot, data_source], 
        chatbot
    )
    clear.click(lambda: None, None, chatbot)

if __name__ == "__main__":
    app.launch(server_port=7860)
