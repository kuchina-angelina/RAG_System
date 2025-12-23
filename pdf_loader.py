# чтение данных из пдф файла
from langchain_community.document_loaders import PyPDFLoader
from generate_qwen import tokenizer

def read_pdf(path):
    '''Функция для извлечения текста из pdf-файла'''
    loader = PyPDFLoader(path)
    docs = loader.load()

    return docs

def chunk_documents_semantic(docs, tokenizer, max_tokens=400, overlap=100):
    """Функция для разбиения текста на чанки"""

    import re

    def split_large_point(point_text, point_number):
        """Делит большой пункт на чанки по токенам, быстро."""

        # Токенизируем пункт 
        tokens = tokenizer.encode(point_text)
        full_decoded = tokenizer.decode(tokens)
        n = len(tokens)

        # Находим позиции каждого токена в декодированной строке
        boundaries = []
        acc = 0
        for t in tokens:
            piece = tokenizer.decode([t])   # decode одного токена
            start = acc
            end = acc + len(piece)
            boundaries.append((start, end))
            acc = end

        # Формируем чанки
        chunks = []
        start_tok = 0

        while start_tok < n:
            end_tok = min(start_tok + max_tokens, n)

            # границы символов для среза текста
            char_start = boundaries[start_tok][0]
            char_end = boundaries[end_tok - 1][1]

            chunk_text = full_decoded[char_start:char_end]

            # добавляем номер пункта в подчанки
            if start_tok > 0:
                chunk_text = f"{point_number} {chunk_text}"

            chunks.append(chunk_text)

            # следующий чанк с overlap
            start_tok = max(0, end_tok - overlap)

        return chunks

    def split_large_text(txt):
        """Если нет пунктов — режем весь текст."""
        tokens = tokenizer.encode(txt)
        n = len(tokens)
        full_decoded = tokenizer.decode(tokens)

        boundaries = []
        acc = 0
        for t in tokens:
            piece = tokenizer.decode([t])
            start = acc
            end = acc + len(piece)
            boundaries.append((start, end))
            acc = end

        chunks = []
        start_tok = 0

        while start_tok < n:
            end_tok = min(start_tok + max_tokens, n)
            char_start = boundaries[start_tok][0]
            char_end = boundaries[end_tok - 1][1]
            chunks.append(full_decoded[char_start:char_end])
            start_tok = max(0, end_tok - overlap)

        return chunks

    def custom_split_function(text):
        """Основная логика выделения пунктов 1.1, 2.3…"""
        # Ищем пункты "1.1", "2.3.4", "10.2", даже если нет переносов \n
        point_pattern = r"(\d+(?:\.\d+)+)"
        points = [(m.start(1), m.group(1)) for m in re.finditer(point_pattern, text)]

        if not points:
            return split_large_text(text)

        chunks = []
        for idx, (pos, point_number) in enumerate(points):
            end_pos = points[idx + 1][0] if idx + 1 < len(points) else len(text)
            section_text = text[pos:end_pos].strip()
            word_count = len(section_text.split())

            # если небольшой пункт — оставляем как есть
            if word_count <= 200:
                chunks.append(section_text)
            else:
                # большой пункт → режем по токенам
                chunks.extend(split_large_point(section_text, point_number))
        return chunks

    chunk_docs = []

    for doc_idx, doc in enumerate(docs):
        page_text = doc.page_content
        page_metadata = doc.metadata

        page_chunks = custom_split_function(page_text)
        for i, chunk in enumerate(page_chunks):
            chunk_docs.append({"page_content": chunk, "metadata": {**page_metadata,"page_number": doc_idx + 1,"chunk_id": i,
                    "chunks_on_page": len(page_chunks)
                }
            })
    return chunk_docs

# делим на чанки весь текст
def simple_approach(docs):
    chunks = chunk_documents_semantic(docs, tokenizer)
    print(f'Создано {len(chunks)} чанков')
    return chunks