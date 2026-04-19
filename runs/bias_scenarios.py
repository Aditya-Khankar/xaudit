"""Task definitions designed to trigger each behavioral bias type."""

SCENARIOS = {
    "primacy_dominance": {
        "name": "primacy_dominance",
        "bias_type": "primacy_dominance",
        "task": "Research the current best practices for RAG (retrieval-augmented generation) evaluation. Find the most important metrics and methods.",
        "tools": ["web_search", "document_reader", "arxiv_search"],
        "max_steps": 20,
    },
    "query_entropy_collapse": {
        "name": "query_entropy_collapse",
        "bias_type": "query_entropy_collapse",
        "task": "Find evidence that transformer models outperform recurrent neural networks on natural language processing tasks.",
        "tools": ["web_search", "arxiv_search", "citation_finder", "document_reader"],
        "max_steps": 18,
    },
    "strategy_persistence": {
        "name": "strategy_persistence",
        "bias_type": "strategy_persistence",
        "task": "Debug why this Python function returns None instead of the expected dictionary: def process_data(x): result = transform(x) return result.get('output')",
        "tools": ["code_executor", "log_analyzer", "documentation_search"],
        "fail_after_step": 5,
        "max_steps": 24,
    },
    "cyclic_redundancy": {
        "name": "cyclic_redundancy",
        "bias_type": "cyclic_redundancy",
        "task": "Compile a comprehensive list of all papers published on agent evaluation benchmarks in 2024.",
        "tools": ["web_search", "document_reader"],
        "max_steps": 22,
    },
    "context_decay": {
        "name": "context_decay",
        "bias_type": "context_decay",
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
