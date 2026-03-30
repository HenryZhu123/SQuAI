import json
import re
import time
import requests


class FalconAgent:
    """
    LLM agent that uses Falcon3-10B-Instruct through the AI71 API.

    This agent implements the same interface as LLMAgent but uses the
    AI71 API to access the Falcon model instead of loading it locally.
    """

    def __init__(self, api_key):
        """
        Initialize the Falcon agent with an API key.

        Args:
            api_key: AI71 API key for accessing the Falcon model
        """
        self.api_key = api_key
        self.api_url = "https://api.ai71.ai/v1/models/falcon-3-10b-instruct/completions"
        print("FalconAgent initialized (using AI71 API)")

    def generate(self, prompt, max_new_tokens=256):
        """
        Generate text using the Falcon model via the AI71 API.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text as a string
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=120
                )

                if response.status_code != 200:
                    payload = {
                        "prompt": f"User: {prompt}\nAssistant:",
                        "max_tokens": max_new_tokens,
                        "temperature": 0.0,
                    }
                    response = requests.post(
                        self.api_url, headers=headers, json=payload, timeout=120
                    )

                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                text_response = ""
                if isinstance(choice.get("message"), dict):
                    text_response = choice["message"].get("content") or ""
                if not text_response:
                    text_response = choice.get("text") or ""

                if not text_response or text_response.strip() in [
                    "",
                    "<|assistant|>",
                ]:
                    return "I don't have enough information to provide a specific answer."

                return text_response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to generate text after {max_retries} attempts: {e}"
                    ) from e
                wait_time = 2**attempt + 1
                print(f"API call failed, retrying in {wait_time}s... ({e!s})")
                time.sleep(wait_time)

        return "I don't have enough information to provide a specific answer."

    def get_log_probs(self, prompt, target_tokens=None):
        if target_tokens is None:
            target_tokens = ["Yes", "No"]
        scores = {}

        for token in target_tokens:
            biased_prompt = (
                f"{prompt}\n\nBased on the above information, I should answer '{token}'."
            )

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                payload = {
                    "prompt": biased_prompt,
                    "max_tokens": 10,
                    "temperature": 0.1,
                }

                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=60
                )
                response.raise_for_status()

                generation = response.json()["choices"][0]["text"].strip()

                if generation.startswith(token):
                    scores[token] = 0.0
                else:
                    scores[token] = -1.0
            except Exception as e:
                print(f"Error getting log probs for token '{token}': {e}")
                scores[token] = -2.0

        return scores

    def batch_process(self, prompts, generate=True, max_new_tokens=256):
        """
        Process a batch of prompts sequentially (same interface as LLMAgent).
        """
        if not prompts:
            return []

        results = []
        for prompt in prompts:
            if generate:
                results.append(self.generate(prompt, max_new_tokens))
            else:
                results.append(self.get_log_probs(prompt, ["Yes", "No"]))

        return results


class DeepSeekAgent:
    """
    LLM agent using DeepSeek's OpenAI-compatible HTTP API (e.g. deepseek-chat / deepseek-reasoner).

    Implements the same interface as LLMAgent / FalconAgent: generate(), get_log_probs(), batch_process().
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, api_key, model="deepseek-chat", base_url=None, timeout=120):
        """
        Args:
            api_key: DeepSeek API key (Bearer token).
            model: Model id, e.g. "deepseek-chat" (V3) or "deepseek-reasoner" (R1).
            base_url: Optional override for OpenAI-compatible base (no trailing /v1 duplication).
            timeout: Request timeout in seconds.
        """
        if not api_key or not str(api_key).strip():
            raise ValueError("DeepSeekAgent requires a non-empty api_key")

        self.api_key = str(api_key).strip()
        self.model = model
        raw_base = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        if raw_base.endswith("/v1"):
            self.base_url = raw_base
        else:
            self.base_url = f"{raw_base}/v1"
        self.timeout = timeout
        self._chat_url = f"{self.base_url}/chat/completions"
        print(f"DeepSeekAgent initialized (model={self.model}, base={self.base_url})")

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _post_chat(self, messages, max_tokens, temperature=0.0, max_retries=3):
        """POST /v1/chat/completions with retries and clear errors."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_err = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self._chat_url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    detail = response.text[:500]
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}: {detail}", response=response
                    )
                body = response.json()
                if "choices" not in body or not body["choices"]:
                    raise ValueError(f"Unexpected API response: {body!r}")
                msg = body["choices"][0].get("message") or {}
                content = msg.get("content")
                if content is None:
                    content = ""
                return str(content).strip()
            except Exception as e:
                last_err = e
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"DeepSeek chat request failed after {max_retries} attempts: {e}"
                    ) from e
                wait_time = 2**attempt + 1
                print(f"DeepSeek API call failed, retrying in {wait_time}s... ({e!s})")
                time.sleep(wait_time)

        raise RuntimeError(f"DeepSeek request failed: {last_err}")

    def generate(self, prompt, max_new_tokens=256):
        """
        Generate completion for a single user prompt (chat format).
        """
        if not prompt or not str(prompt).strip():
            return "I don't have enough information to provide a specific answer."

        messages = [{"role": "user", "content": str(prompt)}]
        try:
            text = self._post_chat(
                messages, max_tokens=max_new_tokens, temperature=0.0
            )
        except Exception as e:
            print(f"DeepSeek generate() error: {e}")
            return "I don't have enough information to provide a specific answer."

        if not text:
            return "I don't have enough information to provide a specific answer."
        return text

    def get_log_probs(self, prompt, target_tokens=None):
        """
        Approximate log-prob style scores for Yes/No (API has no true token logprobs).

        Uses a short chat completion that must start with Yes or No, then maps to scores
        compatible with the adaptive judge bar in run_SQuAI.
        """
        if target_tokens is None:
            target_tokens = ["Yes", "No"]

        system = (
            "You are a strict classifier. Reply starting with exactly one word: "
            '"Yes" or "No" (capital Y or N), then optionally a short reason.'
        )
        user = (
            f"{prompt}\n\n"
            'First word of your answer must be exactly "Yes" or "No".'
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            raw = self._post_chat(messages, max_tokens=64, temperature=0.0)
        except Exception as e:
            print(f"DeepSeek get_log_probs() error: {e}")
            return {t: -2.0 for t in target_tokens}

        first = raw.strip()
        m = re.match(r"^\s*(Yes|No)\b", first, re.IGNORECASE)
        scores = {t: -2.0 for t in target_tokens}
        if m:
            word = m.group(1).capitalize()
            if word == "Yes":
                scores["Yes"] = 0.0
                scores["No"] = -1.5
            else:
                scores["No"] = 0.0
                scores["Yes"] = -1.5
        else:
            upper = first.upper()
            if upper.startswith("Y"):
                scores["Yes"] = -0.5
                scores["No"] = -1.0
            elif upper.startswith("N"):
                scores["No"] = -0.5
                scores["Yes"] = -1.0
            else:
                scores = {t: -1.0 for t in target_tokens}

        for t in target_tokens:
            if t not in scores:
                scores[t] = -2.0
        return scores

    def batch_process(self, prompts, generate=True, max_new_tokens=256):
        """
        Process prompts one by one (same interface as LLMAgent / FalconAgent).
        """
        if not prompts:
            return []

        results = []
        for prompt in prompts:
            if generate:
                results.append(self.generate(prompt, max_new_tokens))
            else:
                results.append(self.get_log_probs(prompt, ["Yes", "No"]))

        return results


def create_four_deepseek_agents(api_key: str, model: str):
    """
    Build four DeepSeekAgent instances (same model id) for 4-agent RAG pipelines.

    Each agent is independent; matches prior pattern of four FalconAgent instances.
    """
    return tuple(DeepSeekAgent(api_key, model=model) for _ in range(4))
