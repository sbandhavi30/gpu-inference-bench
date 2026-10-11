import random
from locust import HttpUser, task, between

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

# (prompt, max_output_tokens) — short-medium I/O targeting ~50-200 input, ~100-400 output tokens
MESSAGES = [
    ("What is gradient descent?", 256),
    ("Explain transformer attention in 3 sentences.", 300),
    ("What is the difference between L1 and L2 regularization?", 256),
    ("How does KV caching improve LLM inference throughput?", 400),
    ("What is tensor parallelism and why does NVLink matter for it?", 400),
    ("Explain the difference between TTFT and TPOT in LLM serving.", 300),
    ("What is vLLM's PagedAttention and what problem does it solve?", 400),
    ("Summarize the key tradeoffs between beam search and sampling.", 350),
    ("What is quantization and how does FP8 affect model quality?", 300),
    ("How does prefix caching reduce TTFT for repeated prompts?", 350),
]


class ChatUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def chat_completion(self):
        prompt, max_tokens = random.choice(MESSAGES)
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            },
            timeout=60,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            else:
                data = response.json()
                if not data.get("choices"):
                    response.failure("No choices in response")
                else:
                    response.success()
