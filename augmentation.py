def prepare_context(top_chunks_rerank):
    """ Функция для формирования контекста перед передачей в модель """

    context_parts = []
    page_numbers = set()

    for item in top_chunks_rerank:
        chunk_data, score = item

        text = chunk_data["page_content"]
        meta = chunk_data["metadata"]

        page_num = meta.get("page_number", "неизвестно")

       
        block = (
            f"[Источник — страница {page_num}]\n"
            f"{text.strip()}"
        )

        context_parts.append(block)
        page_numbers.add(page_num)

    final_context = "\n\n".join(context_parts)

    return final_context, sorted(list(page_numbers))
