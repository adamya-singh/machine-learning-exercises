import os
import pdfplumber
import tiktoken
from openai import OpenAI
import json
import time  # Add time module import

# Initialize the OpenAI client from the environment instead of hard-coding a secret
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Define the prompt (customize as needed)
prompt = """I am fine-tuning an LLM on *Atomic Habits* to make it an expert on the book’s strategies and why they work, so I can integrate it into a habit tracking app as an inbuilt assistant. Below is a section of text from the book. Based on this excerpt, please generate question-answer pairs that will help the LLM learn the key concepts, strategies, and their applications.

**Guidelines for generating question-answer pairs:**

1. Use only the information in the provided excerpt to create questions and answers, without referencing the excerpt itself or assuming knowledge from other parts of the book.
2. **Focus exclusively on generating questions about habit-building strategies, concepts, and their applications.** Avoid questions about the author’s personal story or anecdotes (e.g., do not ask, "What was the initial aftermath of the injury the person experienced?").
3. Include a mix of question types:
    - Questions that test understanding of concepts and strategies (e.g., "What is the two-minute rule?")
    - Questions that ask for explanations of why strategies work (e.g., "Why is starting small effective for building habits?")
    - Questions that ask for practical applications (e.g., "How can I use the two-minute rule to start a new habit?")
4. Frame all questions and answers from the user’s perspective, focusing on their experience and actions, without mentioning 'the text,' 'the author,' or any source-specific terms (e.g., use 'Why should you start small when building habits?' instead of 'Why does the author emphasize starting small?'). Treat the concepts as general knowledge for the user, even though they come from the excerpt.
5. Ensure that answers are accurate, concise, and directly related to the excerpt. Aim for 2-3 sentences per answer, providing clear explanations.
6. Use clear, simple language suitable for a general audience.
7. Write all answers as if the LLM is giving practical, user-focused advice based on universal principles, not as if it's summarizing or interpreting a specific story or source.

Please provide your response in the following JSON format:
{
    "qa_pairs": [
        {
            "question": "Your question here",
            "answer": "Your answer here"
        }
    ]
}
"""

# Set up tiktoken encoding for token counting (using GPT-4's encoding)
encoding = tiktoken.encoding_for_model("gpt-4")

# Function to process the PDF and create chunks
def create_chunks(pdf_path, max_tokens=2000):
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure the page has text
                page_token_count = len(encoding.encode(page_text))
                if current_token_count + page_token_count > max_tokens:
                    if current_chunk:  # Save the current chunk if it exists
                        chunks.append(current_chunk)
                    # Start a new chunk with the current page
                    current_chunk = page_text + "\n"
                    current_token_count = page_token_count
                else:
                    # Add the page text to the current chunk
                    current_chunk += page_text + "\n"
                    current_token_count += page_token_count
    
    # Append any remaining text as the final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Process the PDF
pdf_path = "atomic_habits.pdf"
chunks = create_chunks(pdf_path, max_tokens=2000)

# Initialize a list to store all question-answer pairs
all_qa_pairs = []

# Process each chunk with the OpenAI API
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}")
    chunk_start_time = time.time()  # Start timing
    
    # Combine the prompt with the chunk
    user_message = prompt + chunk
    
    # Make the API call
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # Adjust model as needed (e.g., "gpt-4o", "gpt-3.5-turbo")
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in habit-building strategies designed for integration into a habit tracking app. Your task is to generate question-answer pairs in JSON format that focus exclusively on these strategies, concepts, and their practical applications. Avoid referencing the author's personal story or anecdotes. Frame all questions and answers from the user's perspective, providing concise, actionable advice based on universal principles."},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},  # Enforce JSON output
            max_tokens=2000,  # Limit response size (adjust as needed)
            temperature=0.7  # Control creativity (adjust as needed)
        )
        
        # Parse the JSON response
        response_data = json.loads(response.choices[0].message.content)
        
        # Validate and extract qa_pairs
        if "qa_pairs" in response_data and isinstance(response_data["qa_pairs"], list):
            all_qa_pairs.extend(response_data["qa_pairs"])
            chunk_duration = time.time() - chunk_start_time  # Calculate duration
            print(f"Successfully processed chunk {i + 1}: {len(response_data['qa_pairs'])} QA pairs added (took {chunk_duration:.2f} seconds)")
        else:
            chunk_duration = time.time() - chunk_start_time  # Calculate duration even if there's an error
            print(f"Warning: Unexpected response format in chunk {i + 1} (took {chunk_duration:.2f} seconds)")
            print(f"Response data: {response_data}")
        
    except json.JSONDecodeError as e:
        chunk_duration = time.time() - chunk_start_time  # Calculate duration even if there's an error
        print(f"Error decoding JSON for chunk {i + 1} (took {chunk_duration:.2f} seconds): {e}")
        print(f"Raw response: {response.choices[0].message.content}")
    except Exception as e:
        chunk_duration = time.time() - chunk_start_time  # Calculate duration even if there's an error
        print(f"Error processing chunk {i + 1} (took {chunk_duration:.2f} seconds): {e}")

# Save all QA pairs to a JSON file
output_file = "qa_pairs.json"
with open(output_file, "w") as f:
    json.dump(all_qa_pairs, f, indent=2)

print(f"Done. {len(all_qa_pairs)} QA pairs saved to {output_file}.")
