import requests
import json

url = "http://localhost:8000/search"
query = {"query": "How is the weather?", "top_k": 3}

print(f"Sending query: '{query['query']}'")
try:
    response = requests.post(url, json=query)
    response.raise_for_status()
    data = response.json()
    
    print(f"\nLatency: {data['latency_ms']:.2f} ms")
    print(f"Message: {data['message']}")
    print("\nResults:")
    for i, res in enumerate(data['results']):
        print(f"\n--- Result {i+1} (Score: {res['score']:.4f}) ---")
        print(f"Language: {res['language']}")
        print(f"Child Chunk (Matched): {res['child_chunk']}")
        print(f"Parent Context (Full): {res['parent_context']}")
except Exception as e:
    print(f"Error: {e}")
