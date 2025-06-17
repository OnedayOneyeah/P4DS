import os, re, random, json, shutil
from dotenv import load_dotenv

from openai import OpenAI
from tavily import TavilyClient
from codeinterpreterapi import CodeInterpreterSession, settings

from db import collection_query

# === OpenAI & Tavily 설정 ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

# === Visualization 초기화 (예시) ===
settings.OPENAI_API_KEY=openai_api_key
settings.MODEL = "gpt-4.1-mini"

# === 10개 항목별 초기 Score/Confidence (None) ===
report_card = {
    "Clarity of Vision":       {"score": None, "confidence": None},
    "Team Competency":         {"score": None, "confidence": None},
    "Customer Understanding":  {"score": None, "confidence": None},
    "Product-Market Fit":      {"score": None, "confidence": None},
    "Traction & KPIs":         {"score": None, "confidence": None},
    "Go-to-Market Strategy":   {"score": None, "confidence": None},
    "Financial Readiness":     {"score": None, "confidence": None}
}
num_criteria = len(report_card)

# 평가지표별 정의 -> 이후 평가 시 각 평가지표와 정의 연결해서 사용
criterion_definitions = {
    "Clarity of Vision": (
        "The clarity and coherence of the startup’s long-term goals and mission. It assesses whether the startup has a well-defined direction and purpose."
    ),
    "Team Competency": (
        "The skills, experience, and execution capability of the founding and core team. This includes technical proficiency, role distribution, and adaptability."
    ),
    "Customer Understanding": (
        "The degree to which the startup understands its target customers’ needs, pain points, and behaviors, and how that understanding shapes product or service design."
    ),
    "Product-Market Fit": (
        "How well the product or service satisfies an existing market demand. Includes customer validation, engagement, and retention."
    ),
    "Traction & KPI": (
        "The measurable indicators of progress such as user growth, revenue, engagement, or retention. Demonstrates early signs of market validation or momentum."
    ),
    "Go-to-Market Strategy": (
        "The effectiveness and feasibility of the plan for customer acquisition, distribution, and messaging. Includes sales channels, marketing strategy, and outreach mechanisms."
    ),
    "Financial Readiness": (
        "The current financial state and the ability to sustain or scale the business. Includes budgeting, burn rate awareness, and fundraising preparedness."
    ),
}

CONFIDENCE_THRESHOLD = 80

# 스타트업 stage에 따라 평가지표별 가중치 부여해 overall_score 계산하기 위한 가중치 정보 저장
STAGE_WEIGHTS = {
    "early": {
        "Clarity of Vision": 0.25,
        "Team Competency": 0.25,
        "Customer Understanding": 0.20,
        "Product-Market Fit": 0.15,
        "Traction & KPIs": 0.05,
        "Go-to-Market Strategy": 0.05,
        "Financial Readiness": 0.05
    },
    "growth": {
        "Clarity of Vision": 0.15,
        "Team Competency": 0.15,
        "Customer Understanding": 0.20,
        "Product-Market Fit": 0.20,
        "Traction & KPIs": 0.15,
        "Go-to-Market Strategy": 0.10,
        "Financial Readiness": 0.05
    },
    "scaling": {
        "Clarity of Vision": 0.10,
        "Team Competency": 0.15,
        "Customer Understanding": 0.10,
        "Product-Market Fit": 0.15,
        "Traction & KPIs": 0.20,
        "Go-to-Market Strategy": 0.20,
        "Financial Readiness": 0.10
    }
}

def all_criteria_above_threshold(report: dict, threshold: int) -> bool:
    """
    모든 항목의 confidence가 threshold 이상인지 체크.
    None이면 threshold를 달성했다고 볼 수 없으므로 False.
    """
    for _, v in report.items():
        if v['confidence'] is None or v['confidence'] < threshold:
            return False
    return True

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

# 모든 user input 모아서 스타트업의 단계 LLM으로부터 가져오는 함수 (generate_business_report에서 사용)
def infer_startup_stage_from_input(user_text: str) -> str:
    system_prompt = (
        "You are a startup evaluator AI.\n"
        "Classify the startup's stage based ONLY on the description below.\n\n"
        "Possible stages:\n"
        "- [early]: Idea stage or pre-revenue\n"
        "- [growth]: Has launched product, early user or revenue growth\n"
        "- [scaling]: Proven product-market fit and scaling business\n\n"
        "Return ONLY the stage name wrapped in brackets, e.g., [growth].\n"
        "Do NOT add explanation, just return the label."
    )
    user_prompt = f"Startup description:\n{user_text}\n\nStartup Stage:"
    raw_response = llm_call(system_prompt, user_prompt, temperature=0.0)

    # 정규표현식 기반 파싱
    match = re.search(r"\[(early|growth|scaling)\]", raw_response.lower())
    if match:
        return match.group(1)
    return "unknown"

# 스타트업 stage 고려해서 지표별 가중치 부여해 overall score 계산
def compute_weighted_overall_score(report: dict, stage: str) -> float:
    weights = STAGE_WEIGHTS.get(stage, {})
    total = 0.0
    for criterion, data in report.items():
        score = data.get("score")
        weight = weights.get(criterion, 0)
        if score is not None:
            total += score * weight
    return round(total, 2)

# 스타트업 stage 고려해서 부족한 부분에 대한 피드백 받는 부분
def generate_stage_feedback_text(report: dict, stage: str) -> str:
    """
    현재 스타트업의 평가 점수와 단계(stage)를 바탕으로,
    LLM이 부족한 항목과 그 이유를 요약 피드백으로 생성.

    출력은 자연어 bullet-point 텍스트로 구성됨.
    """

    # 시스템 프롬프트: 역할과 목적 정의
    system_prompt = (
        "You are a startup evaluation assistant.\n"
        "Given a startup's current stage and evaluation scores, "
        "your job is to identify which criteria are relatively weak or risky at this stage.\n\n"
        "Explain briefly and clearly why each low-scoring criterion matters at this stage.\n"
        "Return your feedback in 2~5 bullet points in plain English (no JSON)."
    )

    # 평가지표 요약 텍스트 구성
    def score_str(k, v):
        return f"{k}: Score = {v['score']}/5, Confidence={v['confidence']}%"

    report_summary = "\n".join([
        score_str(k, v) for k, v in report.items()
        if v["score"] is not None and v["confidence"] is not None
    ])

    # 사용자 프롬프트: 점수 + 단계 전달
    user_prompt = (
        f"Startup Stage: {stage}\n\n"
        f"Evaluation Scores:\n{report_summary}\n\n"
        "Based on this information, which areas should the startup improve given its current stage?"
    )

    # LLM 호출
    feedback_text = llm_call(system_prompt, user_prompt, temperature=0.7)

    return feedback_text.strip()

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

    required_keys = list(report_card.keys())
    if len(data.keys()) != num_criteria:
        return None
    for k in required_keys:
        if k not in data:
            return None
        if not isinstance(data[k], dict):
            return None
        if "score" not in data[k] or "confidence" not in data[k]:
            return None

    return data

def update_report_card(report: dict, new_data: dict):
    """
    report_card의 score/confidence를 new_data로 갱신
    """
    for criterion in report.keys():
        report[criterion]["score"] = new_data[criterion]["score"]
        report[criterion]["confidence"] = new_data[criterion]["confidence"]

def search_internet(query: str) -> str:
    response = tavily_client.search(query)
    if response:
        return response
    else:
        return f"[Internet] '{query}'에 대한 검색 결과가 없습니다."

def search_db(query: str, db_type: str) -> str:
    """
    ChromaDB에서 query_texts=[query]로 검색
    db_type에 맞추어 검색
    """  
    results = collection_query(query_texts=[query], n_results=3, db_type=db_type)
    docs = results.get("documents", [[]])[0]
    if docs:
        joined_docs = "\n".join([f"- {d}" for d in docs])
        return f"[DB 검색 결과 - {db_type}]\n{joined_docs}"
    else:
        return f"[DB 검색 결과 - {db_type}] '{query}' 관련 문서가 없습니다."

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

def analyze_and_visualize(all_context: str) -> dict:
    """
    CodeInterpreter로 하여금 3C 관점 (e.g., Company, Customer and Competitors)에서 시장을 분석하고, 이를 시각화 하도록 함.
    """

    # 1) LLM으로부터 visualization을 위한 프롬프트를 받아오기
    system_prompt_1 = (
    "You are an AI assistant tasked with supporting the creation of a business report.\n"
    "Your focus is on evaluating the market perspective within the 3C framework (Customer, Competitor, Company).\n"
    "Your goal is to identify relevant data points that should be explored and recommend prompts for further analysis and data visualization.\n\n"
    "Structure your response using the following format:\n"
    "[Task] A brief description of the specific analysis or insight to be developed.\n\n"
    "[Data] A bullet-point list of the key data elements required to support the analysis.\n\n"
    "[Chart Specs] A bullet-point list of suggested visualizations, including the type of chart and what variables to compare."
    )
    # all_context = "AI Healthcare startup in the U.S.A. In series A, the item is analzing users' biometrics using medical mobiel devices"
    user_prompt_1 = (
    f"Here is the current context and information gathered about a startup company:\n{all_context}\n\n"
    "Based on this context, please identify areas that need to be improved, clarified, or expanded in order to strengthen the market analysis under the 3C framework."
    )
    user_input = llm_call(system_prompt_1, user_prompt_1, temperature=0.7)
    user_input += "\n[Output] A high‑resolution PNG (≈ 1920×1080) suitable for presentations."

    PROMPT_1 = (
        "You are a highly skilled data analyst assigned to address the user's query using reliable, authoritative, and verifiable data sources.\n\n"
        "Your objectives:\n"
        "- Acquire accurate, relevant, and reputable datasets that directly support the user’s inquiry.\n"
        "- Structure the data in a clean, well-formatted pandas DataFrame with appropriate data types and labels.\n"
        "- Conduct a meaningful, insight-driven analysis that provides direct, data-backed answers to the user’s question.\n"
        "- Create compelling, well-labeled visualizations using matplotlib to effectively communicate your findings.\n\n"
        "Visualization Requirements:\n"
        "- Every chart must include:\n"
        "  • A clear and descriptive title.\n"
        "  • Properly labeled axes and legends, where applicable.\n"
        "  • Exact names of companies, institutions, or reports referenced in the data.\n"
        "  • Visible citation of the data source with URL.(e.g., in a corner of the plot).\n\n"
        "Analysis Explanation:\n"
        "- Write a concise and insightful narrative summarizing:\n"
        "  • The methodology and key findings.\n"
        "  • Any observed trends, comparisons, or outliers.\n"
        "  • The broader implications of the analysis.\n"
        "- Provide full URLs to all data sources used in your analysis to ensure traceability and transparency.\n\n"
        "Important Notes:\n"
        "- Only use data from trustworthy, recognized sources (e.g., official government portals, academic institutions, reputable organizations).\n"
        "- Ensure the analysis is self-contained, reproducible, and directly aligned with the user’s request.\n\n"
        "User Query: {user_input}\n"
    )

    # # (2) Tavily search API를 통해서 이미 retrieval한 데이터를 기반으로, 이를 시각화 하는 경우
    # PROMPT_2 = (
    # "You are a data analyst. Based on the dataset below, analyze the topic '{query}' and create a clear visualization. \n"
    # "\t- The dataset is in pandas DataFrame format and contains the following: {response} \n"
    # "\t- Provide a concise explanation of your analysis. \n"
    # "\t- Generate the visualization using matplotlib. \n"
    # "\t- Make sure to include font settings in the code to properly display Korean characters (e.g., use 'Malgun Gothic' on Windows or 'AppleGothic' on macOS). \n"
    # "\t- Also include the setting to prevent minus signs from breaking when rendering Korean text in matplotlib. \n"
    # )

    with CodeInterpreterSession() as session:
        # generate a response based on user input
        response = session.generate_response(
            PROMPT_1.format(user_input = user_input)
        )

    # save visualized images
    response_txt = response.content  # collected_contexts에 추가 가능

    SOURCE_CODEBOX_DIR = ".codebox"

    # 대상 폴더 설정: UI/static/images
    # app.py가 UI 폴더에 있으므로, 상대 경로는 "static/images"가 됩니다.
    DEST_FIGURES_DIR = os.path.join("static", "images") 
    os.makedirs(DEST_FIGURES_DIR, exist_ok=True)

    if os.path.exists(SOURCE_CODEBOX_DIR) and os.path.isdir(SOURCE_CODEBOX_DIR):
        print(f"'{SOURCE_CODEBOX_DIR}' 폴더에서 파일 복사 시도 -> '{DEST_FIGURES_DIR}'") # 로그 변경
        copied_files_count = 0
        for filename in os.listdir(SOURCE_CODEBOX_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                source_file_path = os.path.join(SOURCE_CODEBOX_DIR, filename)
                destination_file_path = os.path.join(DEST_FIGURES_DIR, filename)
                try:
                    shutil.copy2(source_file_path, destination_file_path)
                    print(f"파일 복사됨: {source_file_path} -> {destination_file_path}")
                    copied_files_count += 1
                except Exception as e:
                    print(f"파일 복사 중 오류 ({source_file_path}): {e}")
        if copied_files_count == 0:
            print(f"'{SOURCE_CODEBOX_DIR}' 폴더에서 복사할 이미지 파일을 찾지 못했습니다.")
    else:
        print(f"소스 폴더 '{SOURCE_CODEBOX_DIR}'를 찾을 수 없습니다. CodeInterpreter가 파일을 생성하지 않았을 수 있습니다.")
        
    return response_txt

def generate_db_query(criterion: str, all_context: str, db_type: str) -> str:
    """
    LLM에게 해당 criterion과 컨텍스트 기반으로 DB에 사용할 적절한 검색 쿼리를 생성하게 함.
    """
    db_desc = "real-world Korean startup company examples" if db_type == "startup" else "AI technology trends and statistics"

    system_prompt = (
        "You are an AI assistant generating database search queries for a business report.\n"
        f"The database is focused on {db_desc}.\n"
        "Given the criterion and the context so far, generate a specific search query that will help retrieve relevant documents.\n"
        "Only return the search query. No explanations, no JSON."
    )

    user_prompt = (
        f"Criterion: {criterion}\n\n"
        f"Context:\n{all_context}\n\n"
        "Search query:"
    )

    query = llm_call(system_prompt, user_prompt, temperature=0.3)
    return query.strip()

def generate_user_question_for_criterion(criterion: str, all_context: str) -> str:
    """
    LLM에게:
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

    # print(f"생성된 질문 리스트 ({len(questions_list)}개): {questions_list}")
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
    if action == "AskUser":
        # 1) LLM에게 '무엇을 구체적으로 물어봐야 하는가'를 요청
        full_context_str = "\n".join(collected_contexts)
        question_prompt = generate_user_question_for_criterion(target_criteria, full_context_str)
        return (None, question_prompt)
    
    elif action == "SearchDB_startup":
        full_context_str = "\n".join(collected_contexts)
        query = generate_db_query(target_criteria, full_context_str, db_type="startup")
        db_result = search_db(query, db_type="startup")
        return (f"DB_SUMMARY: {db_result}", None)

    elif action == "SearchDB_report":
        full_context_str = "\n".join(collected_contexts)
        query = generate_db_query(target_criteria, full_context_str, db_type="stanford")
        db_result = search_db(query, db_type="stanford")
        return (f"DB_SUMMARY: {db_result}", None)

    elif action == "SearchInternet":
        # 1) LLM에게 '인터넷에서 검색할 query'를 생성해달라고 요청
        full_context_str = "\n".join(collected_contexts)
        suggested_query = generate_internet_search_query(target_criteria, full_context_str)

        # 2) 실제 인터넷 검색 수행
        net_result = search_internet(suggested_query)

        # 3) 결과를 반환 (검색어 + 검색 결과) (식별자: INTERNET_SUMMARY)
        return (f"INTERNET_SUMMARY: (Internet Search Query: '{suggested_query}')\n{net_result}", None)

    elif action == "RefineOutput":
        # RefineOutput 시, LLM 추가 호출
        full_context_str = "\n".join(collected_contexts)
        refined_text = refine_criterion_output(target_criteria, full_context_str)
        # 식별자: REFINED_OUTPUT
        return (f"REFINED_OUTPUT: (Refined Output about {target_criteria})\n{refined_text}", None)

    elif action == "AnalyzeAndVisualize":
        full_context_str = "\n".join(collected_contexts)
        analysis_with_figure = analyze_and_visualize(full_context_str)
        # 이제 figure_path도 함께 반환
        return (
            f"3C_ANALYSIS: {analysis_with_figure}",
            None
        )

    elif action == "NoActionNeeded":
        return ("(No further actions required.)", None)
    else:
        return (f"(Unknown Action: {action})", None)
 
def ask_llm_for_next_action(report: dict, collected_texts: list, action_history: list) -> dict:
    """
    LLM에게 “다음 액션” + “어느 항목(criterion)인지” + "왜 그 액션을 골랐는지(rationale)"를
    JSON 형식으로 받는다.
    """
    system_prompt = (
        "You are an AI assistant finalizing a startup's business report.\n\n"
        "You have 7 possible actions:\n"
        " 1) AskUser       : Need more specific details from user\n"
        " 2) SearchDB_startup     : Need real-world Korean startup company examples\n"
        " 3) SearchDB_report    : Need AI technology trend, statistics information\n"
        " 4) SearchInternet: Need external info from the web\n"
        " 5) RefineOutput  : Have enough info, want to refine/improve writing\n"
        " 6) AnalyzeAndVisualize : Perform data analysis and generate visualizations using retrieved user and market information.\n"
        " 7) NoActionNeeded: Everything is sufficiently addressed\n\n"
        "When deciding, consider any info gaps or low confidence in the 7 criteria.\n"
        "Also, review the action history and **choose actions that have not yet been tried**.\n" # 수정
        "Avoid repeating the same actions. Especially, if 'AskUser' action was perfrmed before, do not chose 'AskUser' action.\n\n" # 수정
        "Return your decision in JSON with EXACTLY these three keys:\n"
        "  \"criterion\"  -> one of the 10 criteria, or \"None\" if no focus\n"
        "  \"action\"     -> one of [AskUser, SearchDB_startup, SearchDB_report, SearchInternet, RefineOutput, AnalyzeAndVisualize, NoActionNeeded]\n"
        # "  \"action\"     -> one of [SearchDB_startup, SearchDB_report, SearchInternet, AnalyzeAndVisualize, NoActionNeeded]\n"
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

    print("===========", action_history_text)

    user_prompt = (
        f"Current report state:\n{report_summary}\n\n"
        f"Action history so far:\n{action_history_text}\n\n"
        f"Collected contexts:\n{accumulated_context}\n\n"
        "Which single criterion is the biggest priority now, and which action is most appropriate?\n"
        "Also provide a short rationale explaining your choice.\n"
        "Important: Output EXACTLY and ONLY JSON in the following format:\n\n"
        "{\n"
        "  \"criterion\": \"<one_of_the_10_criteria_or_None>\",\n"
        # "  \"action\": \"<SearchDB_startup_or_SearchDB_report_or_SearchInternet_or_AnalyzeAndVisualize_or_NoActionNeeded>\",\n"
        "  \"action\": \"<AskUser_or_SearchDB_startup_or_SearchDB_report_or_SearchInternet_or_RefineOutput_or_AnalyzeAndVisualize_or_NoActionNeeded>\",\n"
        "  \"rationale\": \"<short_reason>\"\n"
        "}\n"
    )

    max_tries = 3
    for attempt in range(max_tries):
        raw = llm_call(system_prompt, user_prompt, temperature=0.0)
        print("[ask_llm_for_next_action] Raw LLM Output:\n", raw)  # 디버그 출력

        try:
            action_data = json.loads(raw.strip())
            print("=========")
            print(action_data)
            print("=========")
            # JSON 키 검사
            if ("criterion" in action_data) and ("action" in action_data) and ("rationale" in action_data):
                valid_actions = ["AskUser", "SearchDB_startup", "SearchDB_report", "SearchInternet", "RefineOutput", "AnalyzeAndVisualize", "NoActionNeeded"]
                # valid_actions = ["SearchDB_startup", "SearchDB_report", "SearchInternet", "RefineOutput", "AnalyzeAndVisualize", "NoActionNeeded"]
                if action_data["action"] in valid_actions:
                    # # TODO 하나씩 시도
                    # action_data["action"]="AnalyzeAndVisualize"
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
    analysis_texts_from_visualization = []
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
        elif c.startswith("3C_ANALYSIS:"):
            analysis_texts_from_visualization.append(c[len("3C_ANALYSIS_TEXT:"):].strip())
        else: general_contexts.append(c)

    # 참고용으로 구조화된 전체 맥락(설명용)
    structured_context = (
        f"**User Input**:\n{''.join(user_inputs)}\n\n"
        f"**DB Summaries**:\n{''.join(db_summaries)}\n\n"
        f"**Internet Summaries**:\n{''.join(net_summaries)}\n\n"
        f"**Refined Outputs**:\n{''.join(refined_outputs)}\n\n"
        f"**3C Analysis (from Visualization step)**:\n{''.join(analysis_texts_from_visualization)}\n\n"
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

    # 누적된 user input 기반으로 스타트업 단계 추측
    startup_stage = infer_startup_stage_from_input(user_only_input_text)

    # report_card에 저장된 각 평가지표를 돌며 지표별 정의를 criterion_definitions와 붙여주기
    criteria_description_block = "\n".join([
    f"{i+1}) \"{k}\": {criterion_definitions.get(k, '')}" for i, k in enumerate(report_card.keys())
    ])
    
    # 원래 아래 프롬프트 내에 평가지표 이름 나열되어 있었는데 위 block으로 채우는 걸로 대체
    system_prompt_2 = (
        "You are an AI assistant that updates the score and confidence of EXACTLY these 10 criteria:\n"
        f"{criteria_description_block}\n\n"
        "IMPORTANT: For scoring and confidence, you must rely ONLY on the user's input below.\n"
        "Ignore any DB or internet data for the actual scoring.\n\n"
        "You MUST ONLY output valid JSON with these EXACT 7 keys. No more, no less, no renaming.\n"
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
    
    # report score 업데이트 이후 overall score 계산
    overall_score = compute_weighted_overall_score(report, startup_stage)

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

    references_md = "## References\n"
    if user_inputs:
        references_md += "\n**User Input**\n"
        for i, ui in enumerate(user_inputs, 1):
            references_md += f"- User Input #{i}: {ui}\n"
    if db_summaries:
        references_md += "\n**DB Summaries (RAG)**\n"
        for i, dbs in enumerate(db_summaries, 1):
            references_md += f"- DB Ref #{i}: {dbs}\n"
    if net_summaries:
        references_md += "\n**Internet Summaries**\n"
        for i, ns in enumerate(net_summaries, 1):
            references_md += f"- Net Ref #{i}: {ns}\n"
    if refined_outputs:
        references_md += "\n**Refined Outputs**\n"
        for i, ro in enumerate(refined_outputs, 1):
            references_md += f"- Refined #{i}: {ro}\n"
    if analysis_texts_from_visualization:
        references_md += "\n**3C Analysis Texts**\n"
        for i, ao in enumerate(analysis_texts_from_visualization, 1): 
            references_md += f"- Analysis Text #{i}: {ao}...\n"

    system_prompt_4 = (
        "You are an AI assistant. You have a preliminary references list from user, DB, and internet.\n"
        "You also have the final business report context.\n"
        "Your task: review all references, and select only those that are directly relevant and supportive of the report.\n"
        "Omit any redundant or off-topic references.\n\n"
        "Output rules:\n"
        "- If the reference includes a URL, use this format:\n"
        "  📍 [Title](URL)\n"
        "  (1–2 line explanation)\n\n"
        "- If the reference does not have a URL, use this format:\n"
        "  📋 **Title (e.g. from user input or DB)**\n"
        "  (1–2 line explanation)\n\n"
        "Leave a blank line between references for readability.\n"
        "Only output the cleaned list of selected references."
    )

    user_prompt_4 = (
        f"Final Report (draft):\n{refined_report_text}\n\n"
        f"Full references:\n{references_md}\n\n"
        "Please filter out any references that are not relevant or are repetitive.\n"
        "Return only the references you think are important for understanding or supporting this business report.\n"
    )

    filtered_references = llm_call(system_prompt_4, user_prompt_4, temperature=0.7)

    user_prompt_3 = (
        f"Updated report card:\n{updated_report_summary}\n\n"
        "Refined text:\n"
        f"{refined_report_text}\n\n"
        "Please produce a comprehensive markdown report with the structure above. "
        "Make sure to include the 3C Analysis and the 7 criteria."
    )

    # LLM으로 최종 report 생성
    final_markdown_report = llm_call(system_prompt_3, user_prompt_3, temperature=0.7)

    # 스타트업 stage 기반 피드백 생성 및 report에 피드백 내용 추가
    stage_feedback_text = generate_stage_feedback_text(report, startup_stage)

    final_markdown_report += "\n\n## Overall Assessment\n"
    final_markdown_report += f"**Startup Stage**: `{startup_stage}`\n"
    final_markdown_report += f"**Weighted Overall Score**: `{overall_score}/5`\n"

    if stage_feedback_text:
        final_markdown_report += "\n**Stage-Based Feedback:**\n"
        final_markdown_report += stage_feedback_text

    final_markdown_report += "\n\n## Relevant References\n"
    final_markdown_report += filtered_references

    return final_markdown_report

def discussion_loop(final_report: str, all_user_inputs: list):
    """
    (새로 추가됨)
    최종 보고서와 유저 입력들을 바탕으로 자유롭게 대화할 수 있는 함수.
    """
    print("\n=== [Discussion Mode] ===")
    print("최종 보고서 및 유저 입력을 바탕으로 자유롭게 대화가 가능합니다.")
    print("종료하려면 'quit'를 입력하세요.\n")

    while True:
        user_ask = input("User: ")
        if user_ask.strip().lower() == "quit":
            print("Discussion 종료.")
            break

        # LLM 호출 (예시)
        system_prompt = (
            "You are an AI assistant discussing the final startup report.\n"
            "You have the final markdown report and all user inputs.\n"
            "Answer any user questions or discuss further improvements.\n"
        )
        user_prompt = (
            f"Final Report:\n{final_report}\n\n"
            f"All User Inputs:\n{all_user_inputs}\n\n"
            f"User's question:\n{user_ask}"
        )
        answer = llm_call(system_prompt, user_prompt, temperature=0.7)
        print(f"Assistant: {answer}\n")

def main_business_report_loop():
    iteration_count = 0
    max_iterations = 10

    # 모든 정보 누적 리스트
    collected_contexts = []
    # 액션 이력 추적 리스트
    action_history = []

    # 1) 사용자 초기 정보
    initial_input = input("초기 스타트업 정보를 간략히 입력하세요: ")
    # 식별자: USER_INPUT (최초 입력)
    collected_contexts.append(f"USER_INPUT: {initial_input}")

    # 2) 첫 보고서 생성
    final_markdown = generate_business_report(report_card, collected_contexts)
    print("[초기 보고서]\n", final_markdown)

    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"\n=== [Iteration {iteration_count}] ===")
        print_report(report_card)

        # (A) 모든 항목이 threshold 이상이면 종료
        if all_criteria_above_threshold(report_card, CONFIDENCE_THRESHOLD):
            print("✅ 모든 항목이 threshold를 달성했습니다. 최종 보고서를 출력합니다.")
            break

        # (B) LLM에게 “다음 액션” 질의
        action_data = ask_llm_for_next_action(report_card, collected_contexts, action_history)
        if action_data is None:
            print("❌ 3회 시도 후에도 액션 JSON 파싱 실패. 루프를 종료합니다.")
            break

        chosen_criterion = action_data["criterion"]
        chosen_action = action_data["action"]
        rationale = action_data["rationale"]

        print(f"[LLM 결정] 다음에 집중할 항목: {chosen_criterion}")
        print(f"[LLM 결정] 선택된 액션: {chosen_action}")
        print(f"[LLM 결정] 사유(rationale): {rationale}")

        if chosen_action == "NoActionNeeded":
            print("LLM이 NoActionNeeded를 선택했습니다. 보고서를 종료합니다.")
            break

        # (C) 액션 수행 → 새 정보 획득
        # if iteration_count == 1:
        #     chosen_action = "AnalyzeAndVisualize"
        action_result = perform_action(chosen_action, chosen_criterion, collected_contexts)
        print(f"[Action 결과] {action_result}")

        # 액션 이력에 추가
        action_log_text = (
            f"Iteration={iteration_count}, "
            f"Action={chosen_action}, "
            f"Criterion={chosen_criterion}, "
            f"Rationale={rationale}, "
            f"Result={action_result}"
        )
        action_history.append(action_log_text)

        # 결과를 context에도 추가
        collected_contexts.append(action_result)

        # (D) LLM으로 보고서 재작성
        final_markdown = generate_business_report(report_card, collected_contexts)
        print("[LLM 최종 보고서(마크다운)]\n", final_markdown)

    # 반복 종료 시점, 최종 출력
    print("\n=== 최종 스타트업 보고서 ===")
    print_report(report_card)
    print("보고서 생성이 완료되었습니다. ✅")

    # (F) 모든 작업 후, discussion_loop() 진입 (새 기능)
    all_user_inputs = []
    for ctx in collected_contexts:
        if ctx.startswith("USER_INPUT:"): 
            all_user_inputs.append(ctx[len("USER_INPUT:"):].strip())
    discussion_loop(final_markdown, all_user_inputs)
