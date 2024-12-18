from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignupForm
from neo4j import GraphDatabase
import numpy as np
import cv2
import json
from keras_facenet import FaceNet
from django.http import JsonResponse, HttpResponse
from datetime import datetime, timezone

NEO4J_URI = 'neo4j+s://abf9edae.databases.neo4j.io'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = '7LbESA2foTba6tOIh5I4V2R9l_65t2PeI1IG1Vq2-8I'

# Initialize FaceNet model
embedder = FaceNet()


class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_user_node(self, username):
        with self.driver.session() as session:
            session.run("CREATE (p:Person {id: $username})", username=username)

    def create_image_node(self, username, embedding):
        creating_at = datetime.now(timezone.utc).isoformat()
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Person {id: $username})
                CREATE (img:Image {id: $username, embedding: $embedding, created_at: $created_at})
                CREATE (p)-[:HAS_IMAGE]->(img)
                """,
                username=username,
                embedding=embedding.tolist(),
                created_at=creating_at
            )

    def get_user_embeddings(self, username):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {id: $username})-[:HAS_IMAGE]->(img:Image)
                RETURN img.embedding AS embedding, img.created_at AS created_at
                """,
                username=username
            )
            return [
                {"embedding": record["embedding"],
                    "created_at": record["created_at"]}
                for record in result
            ]


def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # Hash the password
            user.set_password(form.cleaned_data['password'])
            user.save()
            # Create a Neo4j node for the new user
            neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            try:
                neo4j_handler.create_user_node(user.username)
            finally:
                neo4j_handler.close()
            # Redirect to a success page or login page
            return HttpResponse("Successfully Signed Up!") and redirect('login')
    else:
        form = SignupForm()

    return render(request, 'signup.html', {'form': form})


def loginPage(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('landing')  # Redirect to your landing page
            else:
                return HttpResponse("Username or Password is incorrect!")
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})


def logoutPage(request):
    logout(request)
    return redirect('login')

# Helper Function to Process Image and Generate Embedding


def process_image_and_get_embedding(image_data):
    # Decode the image from memory
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Preprocess the image
    img_resized = cv2.resize(img, (160, 160))
    img_preprocessed = img_resized.astype(
        'float32') / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension
    # Generate embeddings
    embedding = embedder.embeddings(img_array)[0]  # Extract single embedding
    return embedding


@login_required
def landingPage(request):
    neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        # Get embeddings for the logged-in user
        embeddings_data = neo4j_handler.get_user_embeddings(
            request.user.username)
        # Count the number of embeddings
        num_embeddings = len(embeddings_data)
        # Extract creation dates
        embedding_info = [
            {"embedding_id": idx + 1, "created_at": record["created_at"]}
            for idx, record in enumerate(embeddings_data)
        ]
    finally:
        neo4j_handler.close()
    # Prepare context data
    context = {
        "username": request.user.username,
        "num_embeddings": num_embeddings,
        "embedding_info": embedding_info,
        "max_embeddings": 5,  # Define the maximum allowed embeddings
    }
    if request.method == 'POST':
        try:
            # Handle the image sent as a Blob
            image_file = request.FILES.get('image')
            if not image_file:
                return HttpResponse("No image file received.", status=400)
            # Read the image data
            image_bytes = image_file.read()
            '''# Save the raw image data into a txt file
            with open('image_data_2.txt', 'wb') as f:
                f.write(image_bytes)'''
            # Process the image and get embedding
            embedding = process_image_and_get_embedding(image_bytes)
            '''# save the embedding into a txt file
            with open('embedding_data_2.txt', 'wb') as f:
                f.write(embedding.tobytes())'''
            # Save embedding to Neo4j
            neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            try:
                print("Trying to store embedding in Neo4j")
                neo4j_handler.create_image_node(
                    request.user.username, embedding)
                print("Embedding stored in Neo4j")
            finally:
                neo4j_handler.close()
                print("Neo4j connection closed")
            return JsonResponse({"message": "Image processed and embedding stored successfully!"})
        except Exception as e:
            return JsonResponse({"error": f"Error processing image: {e}"}, status=500)
    return render(request, 'landing.html', context)


@login_required
def delete_embedding(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # This is for frontend display purposes
            embedding_id = data.get('embedding_id')
            # This is used for identifying the embedding to delete
            created_at = data.get('created_at')
            if not created_at:
                return JsonResponse({"error": "created_at is required."}, status=400)
            # Connect to Neo4j and delete the embedding based on the created_at timestamp
            neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            try:
                with neo4j_handler.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (p:Person {id: $username})-[:HAS_IMAGE]->(img:Image {created_at: $created_at})
                        DETACH DELETE img
                        RETURN COUNT(img) AS deletedCount
                        """,
                        username=request.user.username,
                        created_at=created_at
                    )
                    deleted_count = result.single()["deletedCount"]
                    if deleted_count == 0:
                        return JsonResponse({"error": "No matching embedding found to delete."}, status=404)
            finally:
                neo4j_handler.close()
            return JsonResponse({"message": "Embedding deleted successfully."})
        except Exception as e:
            return JsonResponse({"error": f"Error deleting embedding: {e}"}, status=500)
    return JsonResponse({"error": "Invalid request method."}, status=400)
