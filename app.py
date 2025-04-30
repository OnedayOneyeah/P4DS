# app.py (전체 코드 - 질문 분리 및 디버깅 강화 v2)
import streamlit as st
import json
import time
import os
import traceback # 오류 추적용

# generate_report.py에서 필요한 함수 및 상수 임포트
from generate_report import (
    client, tavily_client, # API 클라이언트
    INITIAL_REPORT_CARD, # 상수 이름 일관성
    CONFIDENCE_THRESHOLD,
    all_criteria_above_threshold,
    llm_call, # Q&A 등에서 사용
    perform_action, # 반환값이 (text|None, list|None) 버전이어야 함
    ask_llm_for_next_action,
    generate_business_report # 통합 버전
)

# --- Streamlit 앱 기본 설정 ---
st.set_page_config(layout="wide", page_title="AI 스타트업 진단")
st.title("🚀 AI 스타트업 진단 및 보고서 생성")
st.caption("입력된 정보를 바탕으로 AI가 스타트업을 진단하고 보고서를 개선합니다.")

# --- 세션 상태 관리 변수 초기화 ---
default_values = {
    'current_stage': 'initial_form', 'startup_info': {},
    'report_card': json.loads(json.dumps(INITIAL_REPORT_CARD)),
    'collected_contexts': [], 'action_history': [], 'iteration_count': 0,
    'max_iterations': 1,
    # --- AskUser 관련 상태 ---
    'ask_user_pending': False,
    'pending_questions_list': [],     # 질문 리스트
    'current_question_index': 0,      # 현재 질문 인덱스
    'current_criterion_answers': [],  # 현재 답변 임시 저장
    'pending_question_criterion': None, # 질문 대상 기준
    # ------------------------
    'final_report_markdown': None, 'chat_messages': []
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper 함수: 보고서 카드 표시 ---
def format_report_card_for_display(report: dict) -> str:
    lines = ["**📊 현재 진단 현황**"]
    if not isinstance(report, dict): return "**📊 현재 진단 현황**\n\n데이터 오류"
    i = 1
    for criteria, data in report.items():
        score = data.get('score', 'N/A') if isinstance(data, dict) else 'N/A'
        conf = data.get('confidence', 'N/A') if isinstance(data, dict) else 'N/A'
        score_str = score if score is not None else "N/A"
        conf_str = conf if conf is not None else "N/A"
        lines.append(f"{i}. **{criteria}**: {score_str} / 5점 (Confidence: {conf_str}%)")
        i += 1
    lines.append("---")
    return "\n".join(lines)

# --- UI 렌더링 ---

# ==========================
# === 페이지 1: 초기 정보 입력 ===
# ==========================
if st.session_state.current_stage == 'initial_form':
    st.header("1. 스타트업 정보 입력")
    with st.form("startup_form_initial_v2"): # 폼 키 고유성
        st.subheader("기본 정보")
        name = st.text_input("스타트업 이름*", key="st_name_init_v2")
        year = st.number_input("설립 연도*", min_value=1980, max_value=time.localtime().tm_year, step=1, key="st_year_init_v2", value=time.localtime().tm_year -1)
        idea = st.text_area("핵심 아이템/서비스 및 비전*", key="st_idea_init_v2", height=120, placeholder="예: AI 기반 개인 맞춤형 학습 경로 추천 플랫폼")
        st.subheader("팀 및 시장")
        team = st.text_area("주요 팀 구성원의 배경 및 강점*", key="st_team_init_v2", height=100, placeholder="예: AI 박사 2명, 교육 전문가 1명, 개발 경력 5년 이상 3명")
        market = st.text_area("타겟 고객 및 목표 시장의 특징*", key="st_market_init_v2", height=100, placeholder="예: 자기계발에 관심 많은 20-30대 직장인, 국내 약 500만 명 규모 시장")
        problem = st.text_area("해결하려는 고객의 문제점", key="st_problem_init_v2", height=100, placeholder="예: 학습 콘텐츠는 많지만, 자신에게 맞는 학습 순서와 방법을 찾기 어려움")
        st.subheader("현재 상태")
        progress = st.text_area("주요 성과 및 현황*", key="st_progress_init_v2", height=100, placeholder="예: MVP 개발 완료, 베타 테스터 100명 확보 및 피드백 수집 중, 초기 투자 유치 준비")
        competitors = st.text_area("주요 경쟁사 및 차별점", key="st_competitors_init_v2", height=100, placeholder="예: 경쟁사 A (콘텐츠 부족), 경쟁사 B (개인화 약함). 우리는 AI 추천 정확도와 사용자 경험으로 차별화")
        submitted = st.form_submit_button("🤖 AI 진단 시작하기")

        if submitted:
            required_fields = [name, idea, team, market, progress]
            if not all(required_fields):
                st.warning("필수 항목(*)을 모두 입력해주세요.", icon="⚠️")
            else:
                st.session_state.startup_info = {
                    "이름": name, "설립연도": year, "아이템/비전": idea, "팀구성": team,
                    "타겟시장": market, "고객문제": problem, "진행상황": progress, "경쟁사": competitors
                }
                initial_context = f"USER_INPUT: 초기 스타트업 정보\n" + \
                                  "\n".join([f"- {k}: {v}" for k, v in st.session_state.startup_info.items() if v]) # 빈 값 제외

                # 상태 초기화
                st.session_state.collected_contexts = [initial_context]
                st.session_state.action_history = []
                st.session_state.iteration_count = 0
                st.session_state.report_card = json.loads(json.dumps(INITIAL_REPORT_CARD))
                st.session_state.ask_user_pending = False
                st.session_state.pending_questions_list = []
                st.session_state.current_question_index = 0
                st.session_state.current_criterion_answers = []
                st.session_state.pending_question_criterion = None
                st.session_state.final_report_markdown = None
                st.session_state.chat_messages = []
                st.session_state.current_stage = 'generating'
                st.success("정보 입력 완료. AI 진단을 시작합니다...", icon="🚀")
                time.sleep(1.5); st.rerun()

# ===============================
# === 페이지 2: 보고서 생성/개선 루프 ===
# ===============================
elif st.session_state.current_stage == 'generating':
    st.header("2. AI 진단 및 보고서 개선 중...")
    col1, col2 = st.columns([2, 1])
    with col1: action_placeholder = st.empty(); ask_user_container = st.container()
    with col2: progress_bar = st.progress(0.0); report_status_placeholder = st.empty()
    report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))

    # --- 핵심 반복 로직 ---
    if not st.session_state.ask_user_pending: # 사용자 답변 대기 중 아닐 때
        current_iter = st.session_state.iteration_count
        max_iter = st.session_state.max_iterations
        print('확인확인')
        print(current_iter, max_iter)

        # 0. 첫 반복(Iter 0) 처리
        if current_iter == 0:
            with st.spinner("초기 보고서 생성 중..."):
                try:
                    _ = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                    report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
                    st.session_state.action_history.append("Iter=0, Action=InitialReportGeneration - Success")
                    st.session_state.iteration_count = 1
                    progress_bar.progress(1 / max_iter if max_iter > 0 else 0)
                    time.sleep(0.5); st.rerun()
                except Exception as e: st.error(f"초기 보고서 생성 오류: {e}\n{traceback.format_exc()}"); st.stop()

        # 1. 종료 조건 확인 (Iter 1 이상)
        if current_iter > max_iter:
            st.toast(f"최대 반복({max_iter}회) 완료.", icon="🏁")
            st.session_state.current_stage = 'report_ready'; st.rerun()
        # 신뢰도 체크 전 상태 확인
        if all_criteria_above_threshold(st.session_state.report_card, CONFIDENCE_THRESHOLD):
             st.toast("신뢰도 목표 달성!", icon="🎉"); st.session_state.current_stage = 'report_ready'; st.rerun()

        progress_bar.progress(current_iter / max_iter if max_iter > 0 else 0)
        action_placeholder.markdown(f"--- **🔄 반복 {current_iter} / {max_iter}** ---")

        # 2. 다음 액션 결정 (Iter 1 이상)
        with st.spinner(f"반복 {current_iter}: 다음 작업 결정 중..."):
            action_data = ask_llm_for_next_action(
                st.session_state.report_card,
                st.session_state.collected_contexts,
                st.session_state.action_history
            )
        if action_data is None: st.error("다음 작업 결정 실패."); st.session_state.current_stage = 'report_ready'; st.rerun()

        chosen_criterion = action_data.get("criterion"); chosen_action = action_data.get("action"); rationale = action_data.get("rationale", "N/A")
        action_placeholder.markdown(f"**🤖 AI 결정:** 대상=`{chosen_criterion or '전체'}`, 작업=`{chosen_action}`")
        if chosen_action == "NoActionNeeded": st.toast("추가 작업 불필요."); st.session_state.current_stage = 'report_ready'; st.rerun()

        # 3. 액션 수행
        action_log_base = f"Iter={current_iter}, Action={chosen_action}, Criterion={chosen_criterion}"; action_success = False; questions_list_received = None
        with st.spinner(f"작업 수행 중: {chosen_action}..."):
            try:
                 perform_action_return_value = perform_action(chosen_action, chosen_criterion, st.session_state.collected_contexts)
                 if isinstance(perform_action_return_value, tuple) and len(perform_action_return_value) == 2:
                     action_result_text, questions_list_received = perform_action_return_value
                     st.session_state.action_history.append(f"{action_log_base} - Success")
                     action_success = True
                 else:
                     st.warning(f"perform_action이 예상치 못한 값 반환: {perform_action_return_value}", icon="⚠️")
                     st.session_state.action_history.append(f"{action_log_base} - Failed: Invalid return value")
            except Exception as e:
                 st.error(f"{chosen_action} 작업 중 오류: {e}\n{traceback.format_exc()}", icon="🔥")
                 st.session_state.action_history.append(f"{action_log_base} - Failed: {e}")
                 action_result_text, questions_list_received = (f"오류 발생: {e}", None)

        # 4. 액션 결과 처리
        if action_success and questions_list_received is not None: # AskUser 성공
            if isinstance(questions_list_received, list) and questions_list_received:
                st.session_state.ask_user_pending = True
                st.session_state.pending_questions_list = questions_list_received
                st.session_state.current_question_index = 0
                st.session_state.current_criterion_answers = []
                st.session_state.pending_question_criterion = chosen_criterion
                st.toast(f"'{chosen_criterion}' 질문 시작 ({len(questions_list_received)}개).", icon="❓")
                st.rerun() # 리런하여 질문 UI 표시
            else:
                st.warning(f"'{chosen_criterion}' 질문 생성 실패/빈 리스트. 다음 단계로.", icon="⚠️")
                st.session_state.iteration_count += 1; time.sleep(1); st.rerun()

        elif action_success and action_result_text is not None: # 다른 액션 성공
            st.session_state.collected_contexts.append(action_result_text)
            with st.expander(f"⚡ [{chosen_action}] 작업 결과", expanded=False): st.markdown(f"```text\n{action_result_text[:800]}...\n```")
            with st.spinner("보고서 업데이트 중..."):
                try:
                    _ = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                    report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
                    st.session_state.iteration_count += 1
                    time.sleep(0.5); st.rerun()
                except Exception as e: st.error(f"보고서 업데이트 오류: {e}\n{traceback.format_exc()}"); st.stop()
        else: # 액션 실패
            st.warning(f"{chosen_action} 작업 실패. 다음 반복 시도.", icon="⚠️")
            st.session_state.iteration_count += 1; time.sleep(1); st.rerun()


    # --- 사용자 질문/답변 처리 UI ---
    elif st.session_state.ask_user_pending:
        report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
        criterion = st.session_state.pending_question_criterion
        q_list = st.session_state.pending_questions_list
        q_idx = st.session_state.current_question_index
        total_q = len(q_list) if isinstance(q_list, list) else 0

        action_placeholder.markdown(f"--- **🔄 반복 {st.session_state.iteration_count} / {st.session_state.max_iterations}** ---")
        action_placeholder.markdown(f"**✍️ 사용자 답변 입력 ({criterion} 관련)**")

        with ask_user_container:
            if total_q > 0 and 0 <= q_idx < total_q:
                st.info(f"'{criterion}' 질문 {q_idx + 1} / {total_q}")
                try:
                    current_question = q_list[q_idx]
                    st.subheader(f"질문 {q_idx + 1}:")
                    st.markdown(f"> {current_question}")
                except Exception as e:
                    st.error(f"질문 표시 오류: {e}", icon="🔥")
                    st.session_state.ask_user_pending = False; st.session_state.pending_questions_list = []; st.session_state.current_question_index = 0; st.session_state.current_criterion_answers = []; st.session_state.pending_question_criterion = None
                    st.session_state.iteration_count += 1; time.sleep(1); st.rerun(); st.stop()

                form_key = f"answer_form_{criterion}_{q_idx}_{st.session_state.iteration_count}"
                with st.form(form_key):
                    user_answer_key = f"ans_{criterion}_{q_idx}_{st.session_state.iteration_count}"
                    user_answer = st.text_area("답변:", height=150, key=user_answer_key)
                    submitted = st.form_submit_button(f"답변 제출 ({q_idx + 1}/{total_q})")

                    if submitted:
                        if not user_answer.strip(): st.warning("답변 입력 필요", icon="⚠️")
                        else:
                            st.session_state.current_criterion_answers.append({"question": current_question, "answer": user_answer})
                            st.session_state.current_question_index += 1

                            if st.session_state.current_question_index >= total_q: # 마지막 질문 답변 완료
                                st.success("모든 질문 답변 완료!", icon="✅")
                                combined_answers = f"USER_INPUT: ({criterion} 답변 모음)\n" + "\n".join([f"\n[Q] {qa['question']}\n[A] {qa['answer']}\n---" for qa in st.session_state.current_criterion_answers])
                                st.session_state.collected_contexts.append(combined_answers)

                                # 상태 초기화
                                ask_user_pending_before = st.session_state.ask_user_pending
                                st.session_state.ask_user_pending = False; st.session_state.pending_questions_list = []; st.session_state.current_question_index = 0; st.session_state.current_criterion_answers = []; st.session_state.pending_question_criterion = None

                                # 답변 반영 보고서 업데이트
                                with st.spinner("답변 반영 업데이트 중..."):
                                    try:
                                        _ = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                                        report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
                                        st.session_state.iteration_count += 1
                                        time.sleep(1); st.rerun() # 다음 반복으로
                                    except Exception as e: st.error(f"답변 반영 업데이트 오류: {e}\n{traceback.format_exc()}"); st.stop()
                            else: # 다음 질문으로
                                st.rerun() # 다음 질문 표시
            else: # 리스트가 비었거나 인덱스 오류 시 복구
                st.warning("질문 처리 문제 발생. 다음 단계로.", icon="⚠️")
                st.session_state.ask_user_pending = False; st.session_state.pending_questions_list = []; st.session_state.current_question_index = 0; st.session_state.current_criterion_answers = []; st.session_state.pending_question_criterion = None
                st.session_state.iteration_count += 1; time.sleep(1); st.rerun()


# ==============================
# === 페이지 3: 최종 보고서 및 Q&A ===
# ==============================
elif st.session_state.current_stage == 'report_ready' or st.session_state.current_stage == 'qa':
    st.header("3. 최종 AI 진단 보고서")
    if st.session_state.final_report_markdown is None:
        with st.spinner("⏳ 최종 보고서 생성 중..."):
            try:
                 final_md = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                 st.session_state.final_report_markdown = final_md
            except Exception as e:
                 st.error(f"최종 보고서 생성 오류: {e}\n{traceback.format_exc()}");
                 st.session_state.final_report_markdown = f"# 보고서 생성 오류\n\n{e}\n\n마지막 상태:\n{format_report_card_for_display(st.session_state.report_card)}"

    if st.session_state.final_report_markdown:
        st.markdown(st.session_state.final_report_markdown)
        if st.session_state.current_stage == 'report_ready':
             st.session_state.current_stage = 'qa'; st.rerun()
        st.divider()
    else: st.warning("최종 보고서 내용을 표시할 수 없습니다.", icon="⚠️")

    if st.session_state.current_stage == 'qa':
        st.header("4. 보고서 관련 Q&A")
        st.info("생성된 보고서 내용에 대해 AI에게 질문해보세요.")
        for message in st.session_state.chat_messages:
             with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input("보고서에 대해 질문하세요..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty(); message_placeholder.markdown("🤔...")
                try:
                     qa_system_prompt = "당신은 제공된 스타트업 진단 보고서에 대해 답변하는 AI 비서입니다. 보고서 내용을 바탕으로 사용자의 질문에 명확하고 간결하게 답변해주세요. 보고서에 없는 내용은 모른다고 솔직하게 답하세요."
                     qa_context = f"**진단 보고서 내용:**\n{st.session_state.final_report_markdown or '보고서 없음'}\n\n---\n"
                     qa_user_prompt = f"{qa_context}사용자 질문: {prompt}\n\n답변:"
                     response = llm_call(qa_system_prompt, qa_user_prompt, temperature=0.3)
                     message_placeholder.markdown(response)
                     st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                     error_msg = f"답변 생성 오류: {e}"
                     message_placeholder.error(error_msg, icon="🔥")
                     st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

    st.divider()
    if st.button("🔄 다른 스타트업 진단 시작하기"):
         keys_to_reset = list(default_values.keys())
         for key in keys_to_reset:
             if key in st.session_state: del st.session_state[key]
         st.success("새 진단 시작...", icon="✨"); time.sleep(1); st.rerun()

# ==================
# === 예외 상태 처리 ===
# ==================
else:
    st.error("알 수 없는 앱 상태입니다.", icon="🤷")
    st.json(st.session_state.to_dict())
    if st.button("🔄 상태 초기화 및 새로 시작"):
        keys_to_reset = list(default_values.keys())
        for key in keys_to_reset:
            if key in st.session_state: del st.session_state[key]
        st.rerun()