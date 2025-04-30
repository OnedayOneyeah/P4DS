import os
import json
import time
import re # 질문 파싱용
import traceback # 오류 로깅용
from openai import OpenAI
from tavily import TavilyClient
import pandas as pd
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

# === ChromaDB 초기화 (예시) ===
DB_PATH = "my_chromadb_folder"  # 예: "./db" 또는 Google Drive 경로 등
client_chroma = chromadb.PersistentClient(path=DB_PATH)

# 임베딩 함수 설정 (OpenAI Embedding 사용 예시)
openai_embedding = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-ada-002"
)

# 'startup_collection' 컬렉션 준비
collection = client_chroma.get_or_create_collection(
    name="startup_collection",
    embedding_function=openai_embedding
)
# ------------------------------


# === 10개 항목별 초기 Score/Confidence (None) ===
INITIAL_REPORT_CARD = {
    "Clarity of Vision":       {"score": None, "confidence": None},
    "Product-Market Fit":      {"score": None, "confidence": None},
    "Competitive Advantage":   {"score": None, "confidence": None},
    "Team Competency":         {"score": None, "confidence": None},
    "Go-to-Market Strategy":   {"score": None, "confidence": None},
    "Customer Understanding":  {"score": None, "confidence": None},
    "Financial Readiness":     {"score": None, "confidence": None},
    "Scalability Potential":   {"score": None, "confidence": None},
    "Traction & KPIs":         {"score": None, "confidence": None},
    "Fundraising Preparedness":{"score": None, "confidence": None},
}

CONFIDENCE_THRESHOLD = 80

# TODO 추후 수정 필요
def collection_query(query_texts, n_results):
    return {"documents": [["Doc1", "Doc2", "Doc3"]]}

def all_criteria_above_threshold(report: dict, threshold: int) -> bool:
    """
    모든 항목의 confidence가 threshold 이상인지 체크.
    None이면 threshold를 달성했다고 볼 수 없으므로 False.
    """
    for _, v in report.items():
        if v['confidence'] is None or v['confidence'] < threshold:
            return False
    return True

# TODO 삭제
def print_report(report: dict):
    """
    report_card를 사람이 읽기 좋게 프린트.
    None이면 'N/A'로 표시
    """
    print("\n===== 현재 스타트업 진단 보고서 =====")
    i = 1
    for criteria, data in report.items():
        score_str = data['score'] if data['score'] is not None else "N/A"
        conf_str = data['confidence'] if data['confidence'] is not None else "N/A"
        print(f"{i}. {criteria}: {score_str} / 5점 (Confidence: {conf_str}%)")
        i += 1
    print("==================================\n")


def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """
    OpenAI API를 호출해 system+user 프롬프트로부터 답변을 생성.
    """
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # 모델명은 예시 (적절히 교체 가능)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return completion.choices[0].message.content.strip()


def parse_report_card_json(json_str: str) -> dict:
    """
    LLM이 준 JSON을 파싱해, 10개 키가 모두 있는지, 각 value에 "score","confidence"가 있는지 검사.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    required_keys = list(INITIAL_REPORT_CARD.keys())
    if len(data.keys()) != 10:
        return None
    for k in required_keys:
        if k not in data:
            return None
        if not isinstance(data[k], dict):
            return None
        if "score" not in data[k] or "confidence" not in data[k]:
            return None

    return data


def update_report_card(current_report: dict, new_data: dict):
    """
    report_card의 score/confidence를 new_data로 갱신
    """
    updated_report = current_report.copy()
    for criterion in updated_report.keys():
        updated_report[criterion]["score"] = new_data[criterion]["score"]
        updated_report[criterion]["confidence"] = new_data[criterion]["confidence"]


def search_internet(query: str) -> str:
    response = tavily_client.search(query)
    if response:
        return response
    else:
        return f"[Internet] '{query}'에 대한 검색 결과가 없습니다."


def search_db(query: str) -> str:
    """
    ChromaDB에서 query_texts=[query]로 검색
    """
    # mock
    results = collection_query(query_texts=[query], n_results=3)
    docs = results.get("documents", [[]])[0]
    if docs:
        joined_docs = "\n".join([f"- {d}" for d in docs])
        return f"[DB 검색 결과]\n{joined_docs}"
    else:
        return f"[DB 검색 결과] '{query}' 관련 문서가 없습니다."


def refine_criterion_output(criterion: str, all_context: str) -> str:
    """
    'criterion' 항목에 대해 보완할 점을 먼저 추출한 뒤,
    그 보완점을 반영하여 다시 작성된 텍스트를 최종 반환한다.
    """

    # 1) LLM으로부터 개선(보완) 포인트를 먼저 받아온다.
    system_prompt_1 = (
        "You are an AI assistant analyzing a specific section of a startup business report.\n"
        "Your task is to identify any weaknesses or areas for improvement in the text related to this criterion.\n"
        "List them clearly so we can address them in the next step.\n\n"
        "Return these suggested improvements in plain text (e.g., bullet points)."
    )
    user_prompt_1 = (
        f"Criterion to refine: '{criterion}'\n\n"
        f"Here is all the context collected so far:\n{all_context}\n\n"
        "Please list the points or areas that should be improved, clarified, or expanded upon for this criterion."
    )
    improvement_points = llm_call(system_prompt_1, user_prompt_1, temperature=0.7)

    # 2) LLM에게, 위에서 받은 개선 포인트를 반영해 더 깊고 구체적인 텍스트로 다시 작성해달라고 요청한다.
    system_prompt_2 = (
        "You are an AI assistant refining a specific section of a startup business report.\n"
        "You have a list of improvements to address.\n"
        "Use them to produce a revised, more detailed discussion for this criterion, "
        "providing clarity, depth, and actionable insights.\n\n"
        "Return the refined explanation in plain text (no JSON)."
    )
    user_prompt_2 = (
        f"Criterion to refine: '{criterion}'\n\n"
        f"Improvement points:\n{improvement_points}\n\n"
        f"Here is the context again:\n{all_context}\n\n"
        "Incorporate the listed improvements into the final refined text."
    )
    refined_text = llm_call(system_prompt_2, user_prompt_2, temperature=0.7)

    return refined_text


def generate_user_question_for_criterion(criterion: str, all_context: str) -> str:
    """
    LLM에게:ㅊ
      '해당 criterion을 개선하기 위해 사용자에게 어떤 세부 정보를 물어봐야 하는지'
    를 묻는다. LLM이 구체적인 질문 문장을 반환.
    """
    system_prompt = (
        "You are an AI assistant helping gather specific user input. "
        "Given the context and the chosen criterion, generate a short list of specific questions "
        "the user should answer in detail. The output should be plain text (no JSON)."
    )
    user_prompt = (
        f"Criterion of focus: '{criterion}'\n\n"
        f"Context so far:\n{all_context}\n\n"
        "Based on what is missing or uncertain for this criterion, "
        "create a short set of bullet-point questions for the user to answer. "
        "Be as concrete as possible."
    )
    raw_question_text = llm_call(system_prompt, user_prompt, temperature=0.7)

    # LLM 응답 파싱하여 리스트 생성
    questions_list = []
    # 각 줄을 확인하여 질문 패턴 (-, 숫자., *)으로 시작하는지 검사
    lines = raw_question_text.strip().split('\n')
    for line in lines:
        stripped_line = line.strip()
        # 불렛 포인트나 번호 매기기 제거 후 내용 추출
        # 예: "- 질문 내용", "1. 질문 내용", "* 질문 내용" 등 처리
        match = re.match(r'^[\-\*\d]+\.?\s*(.*)', stripped_line)
        if match:
            question_content = match.group(1).strip()
            if question_content: # 내용이 있는 경우만 추가
                 questions_list.append(question_content)
        elif stripped_line: # 패턴이 없지만 내용이 있는 줄도 일단 추가 (LLM이 형식을 안 지킬 경우 대비)
             questions_list.append(stripped_line)

    print(f"생성된 질문 리스트 ({len(questions_list)}개): {questions_list}")
    return questions_list


def generate_internet_search_query(criterion: str, all_context: str) -> str:
    """
    LLM에게: '해당 criterion 관련해서 인터넷에서 어떤 키워드를 검색해야
    필요한 정보를 얻을 수 있는지'를 물어봄.
    """
    system_prompt = (
        "You are an AI assistant that decides the best internet search query "
        "to gather more information about a certain criterion in a startup business report.\n"
        "Return ONLY the recommended search query in plain text (no JSON)."
    )
    user_prompt = (
        f"Criterion of focus: '{criterion}'\n\n"
        f"Context so far:\n{all_context}\n\n"
        "Based on what's missing or uncertain for this criterion, propose a concise search query "
        "that would help gather the most relevant insights or data from the internet."
    )
    suggested_query = llm_call(system_prompt, user_prompt, temperature=0.7)
    return suggested_query.strip()


def perform_action(action: str, target_criteria: str, collected_contexts: list) -> str:
    """
    LLM이 결정한 action을 실제 수행하여 결과 텍스트를 반환.
    결과 텍스트는 그대로 collected_contexts에 추가되어
    이후 보고서 업데이트/분석에 활용된다.

    * 여기서 반환 문자열 앞에 식별자를 붙여줌으로써,
      나중에 generate_business_report에서 구조적으로 활용 가능하도록 함.
    """
    full_context_str = "\n\n".join(collected_contexts)

    if action == "AskUser":
        # 1) LLM에게 '무엇을 구체적으로 물어봐야 하는가'를 요청
        question_prompt = generate_user_question_for_criterion(target_criteria, full_context_str)
        return (None, question_prompt)

    elif action == "SearchDB":
        db_result = search_db(target_criteria)
        # 식별자: DB_SUMMARY
        return (f"DB_SUMMARY: ({target_criteria})\n{db_result}", None)

    elif action == "SearchInternet":
        # 1) LLM에게 '인터넷에서 검색할 query'를 생성해달라고 요청
        suggested_query = generate_internet_search_query(target_criteria, full_context_str)

        # 2) 실제 인터넷 검색 수행
        net_result = search_internet(suggested_query)

        # 3) 결과를 반환 (검색어 + 검색 결과) (식별자: INTERNET_SUMMARY)
        return (f"INTERNET_SUMMARY: (Query: '{suggested_query}')\n{net_result}", None)

    elif action == "RefineOutput":
        # RefineOutput 시, LLM 추가 호출
        refined_text = refine_criterion_output(target_criteria, full_context_str)
        # 식별자: REFINED_OUTPUT
        return (f"REFINED_OUTPUT: ({target_criteria})\n{refined_text}", None)

    elif action == "NoActionNeeded":
        return ("(No further actions required based on current assessment.)", None) # 좀 더 명확한 메시지
    else:
        # 알 수 없는 액션 처리
        return (f"(Unknown Action: {action})", None)


def ask_llm_for_next_action(
    report: dict,
    collected_texts: list,
    action_history: list
) -> dict:
    """
    LLM에게 “다음 액션” + “어느 항목(criterion)인지” + "왜 그 액션을 골랐는지(rationale)"를
    JSON 형식으로 받는다.
    """
    system_prompt = (
        "You are an AI assistant finalizing a startup's business report.\n\n"
        "You have 5 possible actions:\n"
        " 1) AskUser       : Need more specific details from user\n"
        " 2) SearchDB      : Need to query local database (RAG)\n"
        " 3) SearchInternet: Need external info from the web\n"
        " 4) RefineOutput  : Have enough info, want to refine/improve writing\n"
        " 5) NoActionNeeded: Everything is sufficiently addressed\n\n"
        "When deciding, consider any info gaps or low confidence in the 10 criteria.\n\n"
        "Return your decision in JSON with EXACTLY these three keys:\n"
        "  \"criterion\"  -> one of the 10 criteria, or \"None\" if no focus\n"
        "  \"action\"     -> one of [AskUser, SearchDB, SearchInternet, RefineOutput, NoActionNeeded]\n"
        "  \"rationale\"  -> a short sentence explaining why you chose this action.\n\n"
        "No extra keys, no disclaimers, no additional text. ONLY JSON."
    )

    def sc_str(d):
        return f"Score={d['score'] if d['score'] is not None else 'N/A'}, Confidence={d['confidence'] if d['confidence'] is not None else 'N/A'}%"

    report_summary = "\n".join([
        f"{k}: {sc_str(v)}"
        for k, v in report.items()
    ])

    accumulated_context = "\n---\n".join(collected_texts)

    # 액션 히스토리를 텍스트로 합침
    action_history_text = "\n".join([
        f"[{i+1}] {entry}"
        for i, entry in enumerate(action_history)
    ])

    user_prompt = (
        f"Current report state:\n{report_summary}\n\n"
        f"Action history so far:\n{action_history_text}\n\n"
        f"Collected contexts:\n{accumulated_context}\n\n"
        "Which single criterion is the biggest priority now, and which action is most appropriate?\n"
        "Also provide a short rationale explaining your choice.\n"
        "Important: Output EXACTLY and ONLY JSON in the following format:\n\n"
        "{\n"
        "  \"criterion\": \"<one_of_the_10_criteria_or_None>\",\n"
        "  \"action\": \"<AskUser_or_SearchDB_or_SearchInternet_or_RefineOutput_or_NoActionNeeded>\",\n"
        "  \"rationale\": \"<short_reason>\"\n"
        "}\n"
    )

    max_tries = 3
    for attempt in range(max_tries):
        raw = llm_call(system_prompt, user_prompt, temperature=0.0)
        print("[ask_llm_for_next_action] Raw LLM Output:\n", raw)  # 디버그 출력

        try:
            action_data = json.loads(raw.strip())
            # JSON 키 검사
            if ("criterion" in action_data) and ("action" in action_data) and ("rationale" in action_data):
                valid_actions = ["AskUser", "SearchDB", "SearchInternet", "RefineOutput", "NoActionNeeded"]
                if action_data["action"] in valid_actions:
                    return action_data
        except Exception:
            pass

        print(f"⚠️ 액션 JSON 형식 오류(시도 {attempt+1}/{max_tries}), 재시도합니다...")

    return None


def generate_business_report(report: dict, collected_texts: list) -> str:
    """
    단계 요약:
      1) 전체 맥락(DB, 인터넷, 유저 입력 등)을 활용해 '보고서 텍스트(설명 부분)'를 생성
      2) 최종 점수(Score)/신뢰도(Confidence)는 오직 user input만 근거하여 산출
      3) 마크다운 형식으로 최종 보고서 작성

    핵심:
      - 보고서 텍스트(설명)는 DB나 인터넷 요약도 참고해 좀 더 풍부하게 작성한다.
      - 하지만, 10개 기준별 점수는 "user input"만 근거로 한다.
    """

    # -- (1) collected_texts에서 식별자로 분류 --
    user_inputs = []
    db_summaries = []
    net_summaries = []
    refined_outputs = []
    general_contexts = []

    for c in collected_texts:
        if c.startswith("USER_INPUT:"):
            user_inputs.append(c[len("USER_INPUT:"):].strip())
        elif c.startswith("DB_SUMMARY:"):
            db_summaries.append(c[len("DB_SUMMARY:"):].strip())
        elif c.startswith("INTERNET_SUMMARY:"):
            net_summaries.append(c[len("INTERNET_SUMMARY:"):].strip())
        elif c.startswith("REFINED_OUTPUT:"):
            refined_outputs.append(c[len("REFINED_OUTPUT:"):].strip())
        else:
            general_contexts.append(c)

    # 참고용으로 구조화된 전체 맥락(설명용)
    structured_context = (
        f"**User Input**:\n{''.join(user_inputs)}\n\n"
        f"**DB Summaries**:\n{''.join(db_summaries)}\n\n"
        f"**Internet Summaries**:\n{''.join(net_summaries)}\n\n"
        f"**Refined Outputs**:\n{''.join(refined_outputs)}\n\n"
        f"**Other Contexts**:\n{''.join(general_contexts)}"
    )

    # report_card 요약 문자열 (디스플레이용)
    def sc_str(d):
        s = d['score'] if d['score'] is not None else 'N/A'
        c = d['confidence'] if d['confidence'] is not None else 'N/A'
        return f"Score={s}, Confidence={c}%"

    report_summary = "\n".join([
        f"{k}: {sc_str(v)}"
        for k, v in report.items()
    ])

    # ------------------------------------------------------
    # (1) 보고서 설명 텍스트 생성 (DB/인터넷도 참고)
    # ------------------------------------------------------
    system_prompt_1 = (
        "You are an AI assistant that creates business reports for startups.\n"
        "You have access to user input, as well as references from DB and the internet.\n"
        "Use ALL of that context to refine or improve the textual explanation of the report.\n"
        "However, do NOT provide any new scores or confidences here.\n"
        "Just generate the improved discussion/explanation in plain text."
    )
    user_prompt_1 = (
        f"Current report state:\n{report_summary}\n\n"
        "Below is the structured context collected so far:\n"
        f"{structured_context}\n\n"
        "Please provide an updated, more detailed explanation of the business report, "
        "incorporating any relevant insights from the references."
    )
    refined_report_text = llm_call(system_prompt_1, user_prompt_1, temperature=0.7)

    # ------------------------------------------------------
    # (2) 점수(Score)/신뢰도(Confidence) 산출 (오직 user input만 사용)
    # ------------------------------------------------------
    # user input들을 하나로 합침
    user_only_input_text = "\n".join(user_inputs).strip()
    if not user_only_input_text:
        user_only_input_text = "(No user input provided.)"

    system_prompt_2 = (
        "You are an AI assistant that updates the score and confidence of EXACTLY these 10 criteria:\n"
        "1) \"Clarity of Vision\"\n"
        "2) \"Product-Market Fit\"\n"
        "3) \"Competitive Advantage\"\n"
        "4) \"Team Competency\"\n"
        "5) \"Go-to-Market Strategy\"\n"
        "6) \"Customer Understanding\"\n"
        "7) \"Financial Readiness\"\n"
        "8) \"Scalability Potential\"\n"
        "9) \"Traction & KPIs\"\n"
        "10) \"Fundraising Preparedness\"\n\n"
        "IMPORTANT: For scoring and confidence, you must rely ONLY on the user's input below.\n"
        "Ignore any DB or internet data for the actual scoring.\n\n"
        "You MUST ONLY output valid JSON with these EXACT 10 keys. No more, no less, no renaming.\n"
        "Each key => {\"score\": (1~5), \"confidence\": (0~100)}. No extra text."
    )
    user_prompt_2 = (
        "Below is the user's input (the only source for your scoring):\n"
        f"{user_only_input_text}\n\n"
        "Now output ONLY JSON for the updated score/confidence. "
        "Use exactly the 10 keys listed. No extra keys or text."
    )

    new_report_data = None
    max_tries = 3
    for attempt in range(max_tries):
        raw_json_output = llm_call(system_prompt_2, user_prompt_2, temperature=0.0)
        print("[generate_business_report] Raw JSON from LLM:\n", raw_json_output)  # 디버그 로그

        parsed = parse_report_card_json(raw_json_output)
        if parsed is not None:
            new_report_data = parsed
            break
        else:
            print(f"⚠️ JSON 형식 오류(시도 {attempt+1}/{max_tries}), 재요청합니다...")

    if new_report_data:
        update_report_card(report, new_report_data)
    else:
        print("❌ 3회 시도 후에도 JSON 파싱 실패. report_card 업데이트를 건너뜁니다.")

    # ------------------------------------------------------
    # (3) 최종 '마크다운' 형태의 보고서 생성
    # ------------------------------------------------------
    system_prompt_3 = (
        "You are an AI assistant creating a final business report in Markdown format.\n"
        "We have 10 criteria, each with an updated Score and Confidence.\n\n"
        "The final report structure should be:\n"
        "# Startup Diagnostic Report\n"
        "## Introduction\n"
        "(A short overview of the startup's current status)\n\n"
        "## 3C Analysis\n"
        "### Company\n"
        "(Team, resources, culture, etc.)\n\n"
        "### Competitors\n"
        "(Competitive landscape)\n\n"
        "### Customers\n"
        "(Target segments, user needs, insights)\n\n"
        "## Criteria Evaluation\n"
        "For each of the 10 criteria, create a subsection:\n"
        "### {Criterion Name}\n"
        "- Score: X/5\n"
        "- Confidence: Y%\n"
        "- Rationale:\n"
        "  (Short explanation)\n\n"
        "## Conclusion\n"
        "(Summarize key findings and next steps)\n\n"
        "Only output valid Markdown."
    )

    updated_report_summary = "\n".join([
        f"{k}: {sc_str(v)}"
        for k, v in report.items()
    ])

    user_prompt_3 = (
        f"Updated report card:\n{updated_report_summary}\n\n"
        "Refined text:\n"
        f"{refined_report_text}\n\n"
        "Please produce a comprehensive markdown report with the structure above. "
        "Make sure to include the 3C Analysis and the 10 criteria."
    )

    final_markdown_report = llm_call(system_prompt_3, user_prompt_3, temperature=0.7)
    return final_markdown_report