from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
dtype = torch.float32

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_qwen(model, tokenizer, query, context):
    # Формируем промпт
    prompt = (
        "Ты — Qwen, интеллектуальный ассистент. "
        "Отвечай на вопрос, используя ТОЛЬКО информацию из контекста. "
        "Если информации недостаточно, ответь: 'Информации нет.'\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}\nОтвет:"
    )

    # Формируем сообщение в формате chat-template
    messages = [
        {"role": "system", "content": "Ты — Qwen, полезный ассистент от Alibaba Cloud."},
        {"role": "user", "content": prompt}
    ]

    # Преобразуем в токены
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    # Генерация
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Извлекаем только сгенерированную часть
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Декодируем текст
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return response

