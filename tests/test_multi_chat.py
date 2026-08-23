import os
import sys
import json
import shutil
import io
from unittest.mock import MagicMock

# Setup import paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))

import app as flask_app
import upload_files

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=== STARTING MULTI-CHAT WORKSPACE TEST ===")

    # Use a separate test workspace directory to avoid corrupting actual DB
    test_db_dir = "Vector_DB_Chat_Test"
    test_upload_dir = "Uploaded_document_Chat_Test"
    test_metadata_dir = os.path.join(test_db_dir, "chat_metadata")

    # Clean up test directories
    for path in [test_db_dir, test_upload_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)

    # Configure app constants to use test directories
    flask_app.VECTOR_DB_DIR = test_db_dir
    flask_app.METADATA_DIR = test_metadata_dir
    upload_files.UPLOAD_DIR = test_upload_dir
    
    # Re-initialize metadata directory structure
    os.makedirs(test_metadata_dir, exist_ok=True)
    os.makedirs(test_upload_dir, exist_ok=True)

    # Initialize Flask test client
    client = flask_app.app.test_client()

    # Mock the GenerationEngine's generate method to respond dynamically
    def mock_generate_func(prompt_input):
        prompt = prompt_input.get("prompt") if isinstance(prompt_input, dict) else prompt_input
        print(f"\n[Mock Generator] Received Prompt Preview:\n---")
        print(prompt[:300] + "...")
        print("---")
        
        # Check if conversation history is passed
        has_history = "CONVERSATION HISTORY:" in prompt
        
        if "blue" in prompt.lower():
            ans = "Bhupender's favorite color is blue."
            if has_history:
                ans += " (as discussed previously)"
            return {"answer": ans}
        elif "red" in prompt.lower():
            ans = "Bhupender's favorite color is red."
            if has_history:
                ans += " (as discussed previously)"
            return {"answer": ans}
        else:
            return {"answer": "This is a direct chatbot response."}

    flask_app.generator.generate = MagicMock(side_effect=mock_generate_func)

    # --------------------------------------------------------
    # Test 1: List chats initially (should be empty)
    # --------------------------------------------------------
    print("\n[Test 1] Listing chats on startup...")
    res = client.get('/chats')
    data = json.loads(res.data)
    assert res.status_code == 200
    assert len(data["chats"]) == 0
    print("  [Success] Initially no chats exist.")

    # --------------------------------------------------------
    # Test 2: Create Chat A
    # --------------------------------------------------------
    print("\n[Test 2] Creating Chat A...")
    res = client.post('/chats')
    data = json.loads(res.data)
    assert res.status_code == 200
    assert data["success"] is True
    chat_a_id = data["chat"]["chat_id"]
    assert chat_a_id is not None
    assert data["chat"]["name"] == "New Chat"
    print(f"  [Success] Chat A created with ID: {chat_a_id}")

    # --------------------------------------------------------
    # Test 3: Create Chat B
    # --------------------------------------------------------
    print("\n[Test 3] Creating Chat B...")
    res = client.post('/chats')
    data = json.loads(res.data)
    assert res.status_code == 200
    assert data["success"] is True
    chat_b_id = data["chat"]["chat_id"]
    assert chat_b_id is not None
    assert chat_b_id != chat_a_id
    print(f"  [Success] Chat B created with ID: {chat_b_id}")

    # --------------------------------------------------------
    # Test 4: List chats again (should show both)
    # --------------------------------------------------------
    print("\n[Test 4] Listing active chats...")
    res = client.get('/chats')
    data = json.loads(res.data)
    assert len(data["chats"]) == 2
    chat_ids = [c["chat_id"] for c in data["chats"]]
    assert chat_a_id in chat_ids
    assert chat_b_id in chat_ids
    print("  [Success] Both Chat A and Chat B are registered in active list.")

    # --------------------------------------------------------
    # Test 5: Upload doc_a.txt to Chat A
    # --------------------------------------------------------
    print("\n[Test 5] Uploading 'doc_a.txt' to Chat A...")
    upload_data = {
        'chat_id': chat_a_id,
        'files': (io.BytesIO(b"Bhupender's favorite color is blue."), 'doc_a.txt')
    }
    res = client.post('/upload', data=upload_data, content_type='multipart/form-data')
    data = json.loads(res.data)
    assert res.status_code == 200
    assert data["success"] is True
    assert data["ingested_count"] == 1
    
    # Verify folder structure under test upload dir
    chat_a_upload_dir = os.path.join(test_upload_dir, chat_a_id)
    assert os.path.exists(chat_a_upload_dir)
    print("  [Success] doc_a.txt successfully uploaded and ingested under Chat A workspace folder.")

    # --------------------------------------------------------
    # Test 6: Upload doc_b.txt to Chat B
    # --------------------------------------------------------
    print("\n[Test 6] Uploading 'doc_b.txt' to Chat B...")
    upload_data = {
        'chat_id': chat_b_id,
        'files': (io.BytesIO(b"Bhupender's favorite color is red."), 'doc_b.txt')
    }
    res = client.post('/upload', data=upload_data, content_type='multipart/form-data')
    data = json.loads(res.data)
    assert res.status_code == 200
    assert data["success"] is True
    assert data["ingested_count"] == 1
    
    chat_b_upload_dir = os.path.join(test_upload_dir, chat_b_id)
    assert os.path.exists(chat_b_upload_dir)
    print("  [Success] doc_b.txt successfully uploaded and ingested under Chat B workspace folder.")

    # --------------------------------------------------------
    # Test 7: Verify document lists are isolated
    # --------------------------------------------------------
    print("\n[Test 7] Verifying document list separation...")
    # Chat A documents
    res = client.get(f'/documents?chat_id={chat_a_id}')
    data = json.loads(res.data)
    assert len(data["documents"]) == 1
    assert data["documents"][0]["name"] == "doc_a.txt"
    
    # Chat B documents
    res = client.get(f'/documents?chat_id={chat_b_id}')
    data = json.loads(res.data)
    assert len(data["documents"]) == 1
    assert data["documents"][0]["name"] == "doc_b.txt"
    print("  [Success] Documents lists are correctly isolated per chat_id.")

    # --------------------------------------------------------
    # Test 8: Query Chat A (Should retrieve only Chat A context)
    # --------------------------------------------------------
    print("\n[Test 8] Querying Chat A...")
    query_payload = {
        'query': "What is Bhupender's favorite color?",
        'score_threshold': 0.1,  # Low threshold for testing chunks
        'top_k': 1,
        'chat_id': chat_a_id
    }
    res = client.post('/query', json=query_payload)
    data = json.loads(res.data)
    assert res.status_code == 200
    assert data["has_relevant_context"] is True
    # Should find "blue"
    assert "blue" in data["answer"].lower()
    assert "red" not in data["answer"].lower()
    print(f"  [Success] Chat A query returned: '{data['answer']}'")

    # --------------------------------------------------------
    # Test 9: Query Chat B (Should retrieve only Chat B context)
    # --------------------------------------------------------
    print("\n[Test 9] Querying Chat B...")
    query_payload = {
        'query': "What is Bhupender's favorite color?",
        'score_threshold': 0.1,
        'top_k': 1,
        'chat_id': chat_b_id
    }
    res = client.post('/query', json=query_payload)
    data = json.loads(res.data)
    assert res.status_code == 200
    assert data["has_relevant_context"] is True
    # Should find "red"
    assert "red" in data["answer"].lower()
    assert "blue" not in data["answer"].lower()
    print(f"  [Success] Chat B query returned: '{data['answer']}'")

    # --------------------------------------------------------
    # Test 10: Conversational history tracking in Chat A
    # --------------------------------------------------------
    print("\n[Test 10] Testing conversational history tracking in Chat A...")
    # Send a follow-up query to Chat A
    query_payload = {
        'query': "Is it blue?",
        'score_threshold': 0.1,
        'top_k': 1,
        'chat_id': chat_a_id
    }
    res = client.post('/query', json=query_payload)
    data = json.loads(res.data)
    assert res.status_code == 200
    
    # Fetch history of Chat A
    res_hist = client.get(f'/chats/{chat_a_id}/history')
    hist_data = json.loads(res_hist.data)
    assert res_hist.status_code == 200
    history = hist_data["history"]
    
    # We sent 2 queries to Chat A, so history should have 4 messages
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert "favorite color" in history[0]["content"]
    assert history[1]["role"] == "assistant"
    assert "blue" in history[1]["content"]
    assert history[2]["role"] == "user"
    assert "blue" in history[2]["content"]
    assert history[3]["role"] == "assistant"
    
    print("  [Success] Conversational history successfully logged and retrieved.")

    # --------------------------------------------------------
    # Test 11: Auto-renaming chat A
    # --------------------------------------------------------
    print("\n[Test 11] Checking chat auto-rename metadata...")
    res = client.get('/chats')
    chats_list = json.loads(res.data)["chats"]
    chat_a_meta = next(c for c in chats_list if c["chat_id"] == chat_a_id)
    print(f"  Chat A updated name: '{chat_a_meta['name']}'")
    assert "What is Bhupender's" in chat_a_meta["name"]
    print("  [Success] Chat A renamed automatically based on the first query.")

    # --------------------------------------------------------
    # Test 12: Delete Chat A and verify cleanup
    # --------------------------------------------------------
    print("\n[Test 12] Deleting Chat A...")
    res = client.delete(f'/chats/{chat_a_id}')
    data = json.loads(res.data)
    assert res.status_code == 200
    assert data["success"] is True
    
    # Verify metadata clean
    assert not os.path.exists(os.path.join(test_metadata_dir, chat_a_id))
    # Verify files clean
    assert not os.path.exists(chat_a_upload_dir)
    
    # Verify listed chats contains only Chat B
    res = client.get('/chats')
    data = json.loads(res.data)
    assert len(data["chats"]) == 1
    assert data["chats"][0]["chat_id"] == chat_b_id
    print("  [Success] Chat A deletion removed all local workspace resources and metadata.")

    print("\n=== MULTI-CHAT WORKSPACE TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
