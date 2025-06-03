import streamlit as st
import json
import time
import os
import traceback # 오류 추적용

from framework import (
    client, tavily_client, # API 클라이언트
    report_card, # 상수 이름 일관성
    CONFIDENCE_THRESHOLD,
    all_criteria_above_threshold,
    llm_call, # Q&A 등에서 사용
    perform_action, # 반환값이 (text|None, list|None) 버전이어야 함
    ask_llm_for_next_action,
    generate_business_report # 통합 버전
)
 
# --- Streamlit App Basic Settings ---
st.set_page_config(layout="wide", page_title="AI Startup Diagnosis")
st.title("🚀 AI Startup Diagnosis & Report Generation")
st.caption("The AI diagnoses your startup based on the information provided and improves the report.")

# --- Session State Variable Initialization ---
default_values = {
    'current_stage': 'initial_form', 'startup_info': {},
    'report_card': json.loads(json.dumps(report_card)),
    'collected_contexts': [], 'action_history': [], 'iteration_count': 0,
    'max_iterations': 5, # TODO 수정
    # --- AskUser related state ---
    'ask_user_pending': False,
    'pending_questions_list': [],     # List of questions
    'current_question_index': 0,      # Current question index
    'current_criterion_answers': [],  # Temporary storage for current answers
    'pending_question_criterion': None, # Criterion for which questions are being asked
    # ------------------------
    'final_report_markdown': None, 'chat_messages': []
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper function: Display report card ---
def format_report_card_for_display(report: dict) -> str:
    lines = ["**📊 Current Diagnosis Status**"]
    if not isinstance(report, dict): return "**📊 Current Diagnosis Status**\n\nData error"
    i = 1
    for criteria, data in report.items():
        score = data.get('score', 'N/A') if isinstance(data, dict) else 'N/A'
        conf = data.get('confidence', 'N/A') if isinstance(data, dict) else 'N/A'
        score_str = score if score is not None else "N/A"
        conf_str = conf if conf is not None else "N/A"
        lines.append(f"{i}. **{criteria}**: {score_str} / 5 points (Confidence: {conf_str}%)")
        i += 1
    lines.append("---")
    return "\n".join(lines)

# --- UI Rendering ---

# ==========================
# === Page 1: Initial Information Input ===
# ==========================
if st.session_state.current_stage == 'initial_form':
    st.header("1. Enter Startup Information")
    with st.form("startup_form_initial_v2_en"): # Unique form key
        st.subheader("Basic Information")
        name = st.text_input("Startup Name*", key="st_name_init_v2_en")
        year = st.number_input("Year Founded*", min_value=1980, max_value=time.localtime().tm_year, step=1, key="st_year_init_v2_en", value=time.localtime().tm_year -1)
        idea = st.text_area(
            "Core Item/Service & Vision*", 
            key="st_idea_init_v2_en", 
            height=120, 
            placeholder="e.g., AI-based personalized learning path recommendation platform"
        )
        st.subheader("Team & Market")
        team = st.text_area(
            "Key Team Members' Backgrounds & Strengths*", 
            key="st_team_init_v2_en", 
            height=100, 
            placeholder="e.g., 2 AI PhDs, 1 education expert, 3 developers with 5+ years experience"
        )
        market = st.text_area(
            "Target Customers & Market Characteristics*", 
            key="st_market_init_v2_en", 
            height=100, 
            placeholder="e.g., 20-30s professionals interested in self-development, domestic market size approx. 5 million"
        )
        problem = st.text_area(
            "Customer Problem to Solve", 
            key="st_problem_init_v2_en", 
            height=100, 
            placeholder="e.g., Lots of learning content, but difficult to find the right learning order and method"
        )
        st.subheader("Current Status")
        progress = st.text_area(
            "Key Achievements & Current Progress*", 
            key="st_progress_init_v2_en", 
            height=100, 
            placeholder="e.g., MVP development completed, 100 beta testers secured and feedback collected, preparing for seed investment"
        )
        competitors = st.text_area(
            "Main Competitors & Differentiators", 
            key="st_competitors_init_v2_en", 
            height=100, 
            placeholder="e.g., Competitor A (lacks content), Competitor B (weak personalization). We differentiate with AI recommendation accuracy and user experience"
        )
        patents = st.text_area(
            "Patents & Certifications Held", 
            key="st_patents_init_v2_en", 
            height=80, 
            placeholder="e.g., 2 registered domestic patents on AI models, ISO 27001 certified"
        )
        funding_history = st.text_area(
            "Funding History", 
            key="st_funding_init_v2_en", 
            height=80, 
            placeholder="e.g., Seed funding of $500k in 2022 from ABC Ventures, Pre-Series A in progress"
        )
        financials = st.text_area(
            "Revenue, Operating Profit, Net Income", 
            key="st_financials_init_v2_en", 
            height=80, 
            placeholder="e.g., 2024 Revenue: $1.2M, Operating Profit: $200k, Net Income: $150k"
        )
        submitted = st.form_submit_button("🤖 Start AI Diagnosis")

        if submitted:
            required_fields = [name, idea, team, market, progress]
            if not all(required_fields):
                st.warning("Please fill in all required fields (*).", icon="⚠️")
            else:
                st.session_state.startup_info = {
                    "Name": name, "YearFounded": year, "ItemVision": idea, "TeamComposition": team,
                    "TargetMarket": market, "CustomerProblem": problem, "ProgressStatus": progress, "Competitors": competitors
                }
                initial_context = f"USER_INPUT: Initial startup information\n" + \
                                  "\n".join([f"- {k}: {v}" for k, v in st.session_state.startup_info.items() if v]) # Exclude empty values

                # Reset state
                st.session_state.collected_contexts = [initial_context]
                st.session_state.action_history = []
                st.session_state.iteration_count = 0
                st.session_state.report_card = json.loads(json.dumps(report_card))
                st.session_state.ask_user_pending = False
                st.session_state.pending_questions_list = []
                st.session_state.current_question_index = 0
                st.session_state.current_criterion_answers = []
                st.session_state.pending_question_criterion = None
                st.session_state.final_report_markdown = None
                st.session_state.chat_messages = []
                st.session_state.current_stage = 'generating'
                st.success("Information input complete. Starting AI diagnosis...", icon="🚀")
                time.sleep(1.5); st.rerun()

# ===============================
# === Page 2: Report Generation/Improvement Loop ===
# ===============================
elif st.session_state.current_stage == 'generating':
    st.header("2. AI Diagnosis & Report Improvement in Progress...")
    col1, col2 = st.columns([2, 1])
    with col1: action_placeholder = st.empty(); ask_user_container = st.container()
    with col2: progress_bar = st.progress(0.0); report_status_placeholder = st.empty()
    report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))

    # --- Core iteration logic ---
    if not st.session_state.ask_user_pending: # Not waiting for user answer
        current_iter = st.session_state.iteration_count
        max_iter = st.session_state.max_iterations

        # 0. First iteration (Iter 0) processing
        if current_iter == 0:
            with st.spinner("Generating initial report..."):
                try:
                    _ = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                    report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
                    st.session_state.action_history.append("Iter=0, Action=InitialReportGeneration - Success")
                    st.session_state.iteration_count = 1
                    progress_bar.progress(1 / max_iter if max_iter > 0 else 0)
                    time.sleep(0.5); st.rerun()
                except Exception as e: st.error(f"Initial report generation error: {e}\n{traceback.format_exc()}"); st.stop()

        # 1. Check termination conditions (Iter 1 onwards)
        if current_iter > max_iter:
            st.toast(f"Maximum iterations ({max_iter}) completed.", icon="🏁")
            st.session_state.current_stage = 'report_ready'; st.rerun()
        if all_criteria_above_threshold(st.session_state.report_card, CONFIDENCE_THRESHOLD):
             st.toast("Confidence threshold achieved!", icon="🎉"); st.session_state.current_stage = 'report_ready'; st.rerun()

        progress_bar.progress(current_iter / max_iter if max_iter > 0 else 0)
        action_placeholder.markdown(f"--- **🔄 Iteration {current_iter} / {max_iter}** ---")

        # 2. Determine next action (Iter 1 onwards)
        with st.spinner(f"Iteration {current_iter}: Determining next action..."):
            action_data = ask_llm_for_next_action(
                st.session_state.report_card,
                st.session_state.collected_contexts,
                st.session_state.action_history
            )
        if action_data is None: st.error("Failed to determine next action."); st.session_state.current_stage = 'report_ready'; st.rerun()

        chosen_criterion = action_data.get("criterion"); chosen_action = action_data.get("action"); rationale = action_data.get("rationale", "N/A")
        action_placeholder.markdown(f"**🤖 AI Decision:** Target=`{chosen_criterion or 'Overall'}`, Action=`{chosen_action}`, Reason= `{rationale}`")
        if chosen_action == "NoActionNeeded": st.toast("No further action needed."); st.session_state.current_stage = 'report_ready'; st.rerun()

        # 3. Perform action
        action_log_base = f"Iter={current_iter}, Action={chosen_action}, Criterion={chosen_criterion}"; action_success = False; questions_list_received = None
        with st.spinner(f"Performing action: {chosen_action}..."):
            try:
                 perform_action_return_value = perform_action(chosen_action, chosen_criterion, st.session_state.collected_contexts)
                 if isinstance(perform_action_return_value, tuple) and len(perform_action_return_value) == 2:
                     action_result_text, questions_list_received = perform_action_return_value
                     st.session_state.action_history.append(f"{action_log_base} - Success")
                     action_success = True
                 else:
                     st.warning(f"perform_action returned an unexpected value: {perform_action_return_value}", icon="⚠️")
                     st.session_state.action_history.append(f"{action_log_base} - Failed: Invalid return value")
            except Exception as e:
                 st.error(f"Error during {chosen_action}: {e}\n{traceback.format_exc()}", icon="🔥")
                 st.session_state.action_history.append(f"{action_log_base} - Failed: {e}")
                 action_result_text, questions_list_received = (f"Error occurred: {e}", None)

        # 4. Process action result
        if action_success and questions_list_received is not None: # AskUser successful
            if isinstance(questions_list_received, list) and questions_list_received:
                st.session_state.ask_user_pending = True
                st.session_state.pending_questions_list = questions_list_received
                st.session_state.current_question_index = 0
                st.session_state.current_criterion_answers = []
                st.session_state.pending_question_criterion = chosen_criterion
                st.toast(f"Starting questions for '{chosen_criterion}' ({len(questions_list_received)} questions).", icon="❓")
                st.rerun() # Rerun to display question UI
            else:
                st.warning(f"Failed to generate questions for '{chosen_criterion}' or empty list. Proceeding.", icon="⚠️")
                st.session_state.iteration_count += 1; time.sleep(1); st.rerun()
        elif action_success and action_result_text is not None: # Other action successful
            st.session_state.collected_contexts.append(action_result_text)
            # --- MODIFIED PART ---
            with st.expander(f"⚡ [{chosen_action}] Action Result", expanded=False):
                # 이전 코드: st.markdown(f"```text\n{action_result_text[:800]}...\n```")
                # st.text_area를 사용하여 스크롤 가능한 텍스트 박스로 변경
                st.text_area(
                    label="Action Output Details", # text_area에는 label이 필요합니다.
                    value=action_result_text,      # 전체 텍스트를 표시
                    height=300,                    # 원하는 높이로 설정 (픽셀 단위)
                    disabled=True,                 # 읽기 전용으로 설정
                    label_visibility="collapsed"   # label을 화면에 표시하지 않음
                )
            # --- END MODIFIED PART ---
            with st.spinner("Updating report..."):
                try:
                    _ = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                    report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
                    st.session_state.iteration_count += 1
                    time.sleep(0.5); st.rerun()
                except Exception as e: st.error(f"Report update error: {e}\n{traceback.format_exc()}"); st.stop()
        else: # Action failed
            st.warning(f"{chosen_action} action failed. Trying next iteration.", icon="⚠️")
            st.session_state.iteration_count += 1; time.sleep(1); st.rerun()


    # --- User Question/Answer UI ---
    elif st.session_state.ask_user_pending:
        report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
        criterion = st.session_state.pending_question_criterion
        q_list = st.session_state.pending_questions_list
        q_idx = st.session_state.current_question_index
        total_q = len(q_list) if isinstance(q_list, list) else 0

        action_placeholder.markdown(f"--- **🔄 Iteration {st.session_state.iteration_count} / {st.session_state.max_iterations}** ---")
        action_placeholder.markdown(f"**✍️ User Input (regarding {criterion})**")

        with ask_user_container:
            if total_q > 0 and 0 <= q_idx < total_q:
                st.info(f"Question for '{criterion}' {q_idx + 1} / {total_q}")
                try:
                    current_question = q_list[q_idx]
                    st.subheader(f"Question {q_idx + 1}:")
                    st.markdown(f"> {current_question}")
                except Exception as e:
                    st.error(f"Error displaying question: {e}", icon="🔥")
                    st.session_state.ask_user_pending = False; st.session_state.pending_questions_list = []; st.session_state.current_question_index = 0; st.session_state.current_criterion_answers = []; st.session_state.pending_question_criterion = None
                    st.session_state.iteration_count += 1; time.sleep(1); st.rerun(); st.stop()

                form_key = f"answer_form_{criterion}_{q_idx}_{st.session_state.iteration_count}_en"
                with st.form(form_key):
                    user_answer_key = f"ans_{criterion}_{q_idx}_{st.session_state.iteration_count}_en"
                    user_answer = st.text_area("Your Answer:", height=150, key=user_answer_key)
                    submitted = st.form_submit_button(f"Submit Answer ({q_idx + 1}/{total_q})")

                    if submitted:
                        if not user_answer.strip(): st.warning("Please provide an answer.", icon="⚠️")
                        else:
                            st.session_state.current_criterion_answers.append({"question": current_question, "answer": user_answer})
                            st.session_state.current_question_index += 1

                            if st.session_state.current_question_index >= total_q: # Last question answered
                                st.success("All questions answered!", icon="✅")
                                combined_answers = f"USER_INPUT: (Answers for {criterion})\n" + "\n".join([f"\n[Q] {qa['question']}\n[A] {qa['answer']}\n---" for qa in st.session_state.current_criterion_answers])
                                st.session_state.collected_contexts.append(combined_answers)

                                # Reset state
                                st.session_state.ask_user_pending = False; st.session_state.pending_questions_list = []; st.session_state.current_question_index = 0; st.session_state.current_criterion_answers = []; st.session_state.pending_question_criterion = None

                                # Update report with answers
                                with st.spinner("Updating report with your answers..."):
                                    try:
                                        _ = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                                        report_status_placeholder.markdown(format_report_card_for_display(st.session_state.report_card))
                                        st.session_state.iteration_count += 1
                                        time.sleep(1); st.rerun() # To next iteration
                                    except Exception as e: st.error(f"Error updating report with answers: {e}\n{traceback.format_exc()}"); st.stop()
                            else: # Next question
                                st.rerun() # Display next question
            else: # List is empty or index error, recover
                st.warning("Issue with question processing. Proceeding.", icon="⚠️")
                st.session_state.ask_user_pending = False; st.session_state.pending_questions_list = []; st.session_state.current_question_index = 0; st.session_state.current_criterion_answers = []; st.session_state.pending_question_criterion = None
                st.session_state.iteration_count += 1; time.sleep(1); st.rerun()


# ==============================
# === Page 3: Final Report & Q&A ===
# ==============================
elif st.session_state.current_stage == 'report_ready' or st.session_state.current_stage == 'qa':
    st.header("3. Final AI Diagnosis Report & Q&A")

    col_report, col_qa = st.columns([2, 1]) # Left column 2/3, Right column 1/3

    with col_report:
        st.subheader("📋 AI Diagnosis Report")
        if st.session_state.final_report_markdown is None:
            with st.spinner("⏳ Generating final report..."):
                try:
                     final_md = generate_business_report(st.session_state.report_card, st.session_state.collected_contexts)
                     st.session_state.final_report_markdown = final_md
                except Exception as e:
                     st.error(f"Final report generation error: {e}\n{traceback.format_exc()}");
                     st.session_state.final_report_markdown = f"# Report Generation Error\n\n{e}\n\nLast status:\n{format_report_card_for_display(st.session_state.report_card)}"

        if st.session_state.final_report_markdown:
            # To make it scrollable within the column if content is long,
            # you might wrap it in a container with a fixed height,
            # or just let the column handle scrolling.
            # For simplicity, we'll let the column handle natural scrolling.
            st.markdown(st.session_state.final_report_markdown)
            if st.session_state.current_stage == 'report_ready':
                 st.session_state.current_stage = 'qa'; st.rerun() 
        else:
            st.warning("Cannot display final report content.", icon="⚠️")

        # app.py의 적절한 위치에 테스트 코드 추가
        st.subheader("Generated Images")
        imgs = []
        actual_image_folder_on_disk = os.path.join("static", "images") 
        if os.path.exists(actual_image_folder_on_disk) and os.path.isdir(actual_image_folder_on_disk):
            for filename in sorted(os.listdir(actual_image_folder_on_disk)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    imgs.append((os.path.join(actual_image_folder_on_disk, filename), filename.split('.')[0]))
        for i in imgs:       
            st.image(i[0], caption=f"{i[1]}")

    with col_qa:
        st.subheader("💬 Report Q&A")
        if st.session_state.current_stage == 'qa': # Only show Q&A if stage is 'qa'
            st.info("Ask the AI questions about the generated report or your previous inputs.")

            chat_display_container = st.container()

            with chat_display_container:
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Q&A 입력받기 (streamlit 1.25 이상)
            if prompt := st.chat_input("Ask about the report or your inputs..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with chat_display_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown("🤔 Thinking...")

                        # NEW: 시스템 프롬프트, 유저 입력 전체를 맥락으로 활용
                        qa_system_prompt = (
                            "You are an AI assistant discussing the final startup report with the user.\n"
                            "You have access to the final markdown report and all user-provided inputs and answers from previous steps.\n"
                            "Base your answers and suggestions on this information.\n"
                            "If you do not know or the information is insufficient, say so honestly."
                        )
                        qa_context = (
                            f"=== Final Diagnosis Report ===\n"
                            f"{st.session_state.final_report_markdown or 'No report available'}\n\n"
                            f"=== All User Inputs ===\n"
                            f"{st.session_state.collected_contexts}\n\n"
                        )
                        qa_user_prompt = (
                            f"{qa_context}"
                            f"User's question:\n{prompt}\n\n"
                            f"Answer (be specific and grounded in the above context):"
                        )
                        try:
                            response = llm_call(qa_system_prompt, qa_user_prompt, temperature=0.5)
                            message_placeholder.markdown(response)
                            st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Answer generation error: {e}"
                            message_placeholder.error(error_msg, icon="🔥")
                            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                st.rerun()  # To refresh the chat with the new message

    # This button should be outside and below the columns
    st.divider()
    if st.button("🔄 Start New Diagnosis"):
         keys_to_reset = list(default_values.keys())
         for key in keys_to_reset:
             if key in st.session_state: del st.session_state[key]
         st.success("Starting new diagnosis...", icon="✨"); time.sleep(1); st.rerun()

# ==================
# === Exception Handling for Unknown State ===
# ==================
else:
    st.error("Unknown application state.", icon="🤷")
    st.json(st.session_state.to_dict())
    if st.button("🔄 Reset State and Start Over"):
        keys_to_reset = list(default_values.keys())
        for key in keys_to_reset:
            if key in st.session_state: del st.session_state[key]
        st.rerun()


