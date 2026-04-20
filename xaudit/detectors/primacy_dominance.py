"""
Primacy Dominance Detector

Detects over-reliance on first retrieval using Wasserstein-1 distance
(Earth Mover's Distance) from optimal transport theory.

Measures how far the agent's information usage distribution deviates
from uniform — a rational agent distributes attention across all
retrievals proportionally to their relevance.

Method: Wasserstein-1 distance between observed retrieval weights and uniform baseline distribution.
Field:  Optimal transport theory, information geometry.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from xaudit.detectors.base import BaseDetector, DetectorResult
from xaudit.recorder.trace_recorder import AgentTrace

# Defense-in-depth: truncate before TF-IDF to prevent memory spike
# even if input bypasses CLI validators (e.g. Python API usage)
MAX_TFIDF_INPUT = 5000


class PrimacyDominanceDetector(BaseDetector):
    name = "primacy_dominance"
    version = "1.0.0"
    threshold = 0.35

    def detect(self, trace: AgentTrace) -> DetectorResult:
        retrievals = trace.retrievals
        llm_calls = trace.llm_calls

        # Need at least 2 retrievals and 1 LLM output to compute
        if len(retrievals) < 2:
            return self._insufficient(
                "Fewer than 2 retrieval events — primacy dominance is not computable."
            )
        if not llm_calls:
            return self._insufficient("No LLM output events found.")

        # Use final LLM call as the answer
        final_output = str(llm_calls[-1].output)[:MAX_TFIDF_INPUT]
        if not final_output.strip():
            return self._insufficient("Final LLM output is empty.")

        # Build text corpus: all retrieval outputs + final answer
        retrieval_texts = []
        for r in retrievals:
            text = str(r.output).strip()
            if text:
                retrieval_texts.append(text[:MAX_TFIDF_INPUT])

        if len(retrieval_texts) < 2:
            return self._insufficient(
                "Less than 2 non-empty retrieval outputs."
            )

        docs = retrieval_texts + [final_output]

        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform(docs)
        except ValueError:
            # TF-IDF fails on very short or empty documents
            return self._insufficient(
                "Retrieval outputs too short for TF-IDF analysis."
            )

        answer_vec = tfidf_matrix[-1]
        
        weights = []
        for i in range(len(retrieval_texts)):
            sim = float(cosine_similarity(tfidf_matrix[i], answer_vec)[0][0])
            weights.append(sim)
            
        total_weight = sum(weights)
        if total_weight == 0:
            observed = [1.0 / len(weights)] * len(weights)
        else:
            observed = [w / total_weight for w in weights]
            
        n_retrievals = len(weights)
        uniform = [1.0 / n_retrievals] * n_retrievals

        # Use indices to represent distance between retrieval positions
        indices = np.arange(n_retrievals)
        score = float(wasserstein_distance(indices, indices, observed, uniform))

        first_sim = weights[0]
        avg_later_sim = float(np.mean(weights[1:])) if len(weights) > 1 else 0.0

        # Must be skewed toward the first retrieval specifically
        is_primacy = first_sim > avg_later_sim

        return DetectorResult(
            detector_name=self.name,
            detected=(score > self.threshold) and is_primacy,
            score=round(float(score), 3),
            threshold=self.threshold,
            evidence={
                "first_retrieval_similarity_to_answer": round(first_sim, 3),
                "avg_later_retrieval_similarity": round(avg_later_sim, 3),
                "first_retrieval_step": retrievals[0].step_index,
                "answer_step": llm_calls[-1].step_index,
                "total_retrievals": len(retrievals),
            },
            interpretation=(
                f"Answer was {first_sim:.0%} similar to first retrieval. "
                f"Later retrievals averaged {avg_later_sim:.0%} similarity. "
                + (
                    "Primacy dominance pattern detected."
                    if (score > self.threshold) and is_primacy
                    else "No significant primacy dominance detected."
                )
            ),
            detector_version=self.version,
        )
