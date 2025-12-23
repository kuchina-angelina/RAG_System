import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid


# Загрузка модели один раз при импорте
embedding_model = SentenceTransformer("cointegrated/rubert-tiny2")

def create_embeddings(embedding_model, all_text_chunks):
  # функция для преобразования текста в эмбеддинги
  all_embeddings = []
  for text_chunk in all_text_chunks:
    embedding = embedding_model.encode(
        text_chunk['page_content'],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    all_embeddings.append(embedding)
  all_embeddings = np.array(all_embeddings, dtype='float32')

# на выходе список эмбеддингов
  return all_embeddings

# Инициализация клиента и коллекции
client = chromadb.Client()
collection = client.create_collection(name="my_collection")

def add_doc_to_db(all_text_chunks, all_embeddings, collection, filename=None):
    """Добавляет новые чанки в существующую коллекцию без перезаписи старых"""

    # Генерация уникальных идентификаторов
    ids = [f"{uuid.uuid4()}" for _ in all_text_chunks]

    # Извлекаем тексты и метаданные
    all_docs = [chunk["page_content"] for chunk in all_text_chunks]
    all_metadatas = [chunk["metadata"] for chunk in all_text_chunks]

    # Добавляем новые документы в существующую коллекцию
    collection.add(
        ids=ids,
        documents=all_docs,
        embeddings=all_embeddings.tolist(),
        metadatas=all_metadatas
    )

    print(f"Добавлено документов: {len(all_text_chunks)}")
    print(f"Всего документов в коллекции: {collection.count()}")

    return 