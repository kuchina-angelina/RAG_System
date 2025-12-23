from sentence_transformers import CrossEncoder
reranker_model = CrossEncoder("DiTy/cross-encoder-russian-msmarco", max_length=512)

def rerank(reranker_model, query, top_texts_semantic_search, top_n=3):
    """Переупорядочивает результаты с помощью Cross-Encoder"""

    # Подготавливаем список документов (только текст)
    documents = [chunk_text if isinstance(chunk_text, str) else chunk_text['page_content']
                 for chunk_text in top_texts_semantic_search]

    # Получаем ранжированные результаты
    results = reranker_model.rank(query, documents)

    # Собираем результаты (в порядке релевантности)
    reranked_results = []
    for item in results[:top_n]:
        idx = item['corpus_id']
        score = item['score']
        reranked_results.append((top_texts_semantic_search[idx], score))

    return reranked_results




