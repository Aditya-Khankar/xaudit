"""Detects first-retrieval dominance — agent over-relies on its first
retrieval result regardless of contradicting information found later."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace

# Defense-in-depth: truncate before TF-IDF to prevent memory spike
# even if input bypasses CLI validators (e.g. Python API usage)
MAX_TFIDF_INPUT = 5000


class AnchoringDetector(BaseDetector):
    name = "anchoring"
    version = "1.0.0"
    threshold = 0.60

    def detect(self, trace: AgentTrace) -> DetectorResult:
        retrievals = trace.retrievals
        llm_calls = trace.llm_calls

        # Need at least 2 retrievals and 1 LLM output to compute
        if len(retrievals) < 2:
            return self._insufficient(
                "Less than 2 retrieval events — anchoring not computable."
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
        first_sim = float(
            cosine_similarity(tfidf_matrix[0], answer_vec)[0][0]
        )
        later_sims = [
            float(cosine_similarity(tfidf_matrix[i], answer_vec)[0][0])
            for i in range(1, len(retrieval_texts))
        ]
        avg_later_sim = float(np.mean(later_sims)) if later_sims else 0.0

        score = self._clamp(first_sim)

        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "first_retrieval_similarity_to_answer": round(first_sim, 3),
                "avg_later_retrieval_similarity": round(avg_later_sim, 3),
                "first_retrieval_step": retrievals[0].step_index,
                "answer_step": llm_calls[-1].step_index,
                "total_retrievals": len(retrievals),
            },
            interpretation=(
                f"Answer was {score:.0%} similar to first retrieval. "
                f"Later retrievals averaged {avg_later_sim:.0%} similarity. "
                + (
                    "Anchoring pattern detected."
                    if score >= self.threshold
                    else "No significant anchoring detected."
                )
            ),
            detector_version=self.version,
        )
