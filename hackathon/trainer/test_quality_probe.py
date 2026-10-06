from quality_probe import query_for, retrieve


def test_retrieval_excludes_gold_and_abstains_without_evidence():
    row = {"messages": [{"role": "user", "content": "How did petroleum form?"}, {"role": "assistant", "content": "SECRET reference answer"}]}
    query = query_for(row)
    assert "SECRET" not in query
    docs = [{"id": "oil", "title": "Petroleum formation", "text": "Petroleum can form from buried organisms."}]
    assert retrieve(query, docs)[0]["id"] == "oil"
    assert retrieve("Translate a greeting into French", docs) == []


def test_rag_budget_preserves_question():
    from advisor import retrieval_evaluation_prompt
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(m["content"] for m in messages)
        def encode(self, text, **kwargs):
            return list(text)
    row={"prompt":"Explain petroleum formation. QUESTION END", "completion":"SECRET"}
    docs=[{"id":"oil","title":"Petroleum formation","text":"petroleum formation " * 1000}]
    prompt, reference, sources = retrieval_evaluation_prompt(row, Tokenizer(), "Be clear", docs, max_length=500)
    assert len(prompt)<=500
    assert prompt.endswith("QUESTION END")
    assert "SECRET" not in prompt
    assert reference=="SECRET"
    assert sources
