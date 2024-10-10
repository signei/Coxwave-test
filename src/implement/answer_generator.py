from openai import OpenAI

def generate_answer(query, similar_questions, faq_data, conversation_history, config):
    context = "\n".join([f"Q: {q}\nA: {faq_data[q]}" for q, _ in similar_questions])
    
    recent_conversation = "\n".join(conversation_history[-4:])
    
    client = OpenAI(api_key=config['openai_api_key'])
    
    prompt = f"""네이버 스마트스토어 FAQ 도우미입니다. 아래의 사용자 질문에 대해 제공된 FAQ 정보를 정확하게 참조하여 답변해 주세요.
    - 질문과 관련된 FAQ 정보를 최대한 활용하여 정확한 답변을 제공하세요.
    - 사용자의 상황(예: 나이)을 고려하여 맞춤형 답변을 제공하세요.
    - FAQ에 명시된 정보와 모순되는 답변을 하지 마세요.
    - 확실하지 않은 정보는 제공하지 마세요.
    - 질문과 관련이 없는 내용은 답변에 포함하지 마세요.
    - 질문에 대한 직접적이고 명확한 답변을 제공하세요.
    - 사용자의 입력을 그대로 반복하지 마세요.
    - 만약 사용자의 질문이 FAQ와 직접적인 관련이 없다면, 네이버 스마트스토어와 관련된 질문을 해달라고 정중히 요청하세요.
    - 미성년자(만 14세 이상 ~ 만 19세 미만)의 판매 회원 등록에 대한 질문이 있다면, 필요한 서류와 절차를 상세히 안내해 주세요.

    FAQ 정보:
    {context}

    최근 대화:
    {recent_conversation}

    사용자 질문: {query}
    답변:"""

    response = client.chat.completions.create(
        model=config['model_name'],
        messages=[
            {"role": "system", "content": "당신은 네이버 스마트스토어에 대한 질문에 답변하는 도우미입니다. FAQ 정보를 정확히 참조하여 답변하세요. 사용자의 상황을 고려한 맞춤형 답변을 제공하세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config['max_tokens']
    )

    return response.choices[0].message.content.strip()