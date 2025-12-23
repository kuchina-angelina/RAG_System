def semantic_search(embedding_model, collection, query, k=6):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)

    # Извлекаем документы и метаданные
    top_docs = results.get("documents", [[]])[0]
    top_metas = results.get("metadatas", [[]])[0]
    print('3')
    # Формируем результат в нужном формате
    top_embeddings = [
        {"page_content": doc, "metadata": meta}
        for doc, meta in zip(top_docs, top_metas)
    ]

    return top_embeddings

