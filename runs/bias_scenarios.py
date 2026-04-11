"""Task definitions designed to trigger each behavioral bias type."""

SCENARIOS = {
    "anchoring": {
        "name": "anchoring",
        "bias_type": "anchoring",
        "task": "Research the current best practices for RAG (retrieval-augmented generation) evaluation. Find the most important metrics and methods.",
        "tools": ["web_search", "document_reader", "arxiv_search"],
        "max_steps": 20,
    },
    "confirmation": {
        "name": "confirmation",
        "bias_type": "confirmation",
        "task": "Find evidence that transformer models outperform recurrent neural networks on natural language processing tasks.",
        "tools": ["web_search", "arxiv_search", "citation_finder", "document_reader"],
        "max_steps": 18,
    },
    "sunk_cost": {
        "name": "sunk_cost",
        "bias_type": "sunk_cost",
        "task": "Debug why this Python function returns None instead of the expected dictionary: def process_data(x): result = transform(x) return result.get('output')",
        "tools": ["code_executor", "log_analyzer", "documentation_search"],
        "fail_after_step": 5,
        "max_steps": 24,
    },
    "loop": {
        "name": "loop",
        "bias_type": "loop",
        "task": "Compile a comprehensive list of all papers published on agent evaluation benchmarks in 2024.",
        "tools": ["web_search", "document_reader"],
        "max_steps": 22,
    },
    "degradation": {
        "name": "degradation",
        "bias_type": "degradation",
        "task": "Create a 15-step research agenda covering all major open problems in large language model evaluation.",
        "tools": ["web_search", "document_reader", "arxiv_search", "note_taker"],
        "max_steps": 30,
    },
    "clean": {
        "name": "clean",
        "bias_type": None,
        "task": "Find the current exchange rate between USD and EUR and summarize recent trends.",
        "tools": ["web_search", "financial_data", "calculator"],
        "max_steps": 12,
    },
}
