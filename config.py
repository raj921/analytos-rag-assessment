import os

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
SEMANTIC_CONFIG = os.environ["AZURE_SEARCH_SEMANTIC_CONFIG"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

TOP_K = 5
VECTOR_K = 5
RERANKER_MIN_SCORE = 2.0
HIGH_CONFIDENCE_SCORE = 3.0
HISTORY_MAX_TURNS = 6
CACHE_TTL_SECONDS = 600
CACHE_MAX_ITEMS = 256

ROLE_TITLES = {
    "hr": {"Benefits.pdf", "LeavePolicy.pdf"},
    "it": {"PasswordPolicy.docx", "VPNGuide.pdf"},
    "finance": {"TravelPolicy.docx", "ExpensePolicy.pdf"},
    "legal": {"NDA.docx", "VendorContract.pdf"},
    "sales": {"Discounts.xlsx", "Pricing2025.pdf", "Pricing2026.pdf"},
}

REFUSAL_TEXT = "I don't have enough information in the knowledge base to answer that."

BASELINE = os.environ.get("RAG_BASELINE", "").lower() in ("1", "true", "yes")
