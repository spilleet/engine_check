import os
import pathlib
import time
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

ROOT = pathlib.Path(__file__).parent.parent

# --- 1. 가상 데이터베이스 및 로그 도구 파일 정의 ---
def _get_mock_flight_schedule(unit_id: int) -> int:
    return (unit_id * 7 + 13) % 25 + 2

def _get_mock_contact(unit_id: int) -> dict:
    zones = ["A", "B", "C"]
    zone = zones[unit_id % 3]
    managers = ["김정비 과장", "이대리 대리", "박팀장 팀장"]
    manager = managers[unit_id % 3]
    phone = f"010-9999-{1000 + unit_id:04d}"
    slack_channel = f"#mro-zone-{zone.lower()}"
    return {
        "manager": manager,
        "phone": phone,
        "slack_channel": slack_channel,
        "zone": zone
    }

def _log_alert(message: str):
    log_dir = ROOT / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "alerts.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# --- 2. LangChain 에이전트용 Tools 정의 ---

@tool
def check_current_time_context(tick: int) -> str:
    """현재 시뮬레이션 틱(tick)을 기준으로 야간 비근무 시간대(22:00 ~ 08:00)인지 주간 교대 시간대인지 판별합니다."""
    minutes_in_day = tick % 1440
    hours = minutes_in_day // 60
    mins = minutes_in_day % 60
    time_str = f"{hours:02d}:{mins:02d}"
    
    if minutes_in_day >= 1320 or minutes_in_day < 480:
        return f"현재 시각 {time_str} [NIGHT_OFF_HOURS] (야간 비출근 시간대 - 비상 관리자 숙면 중. 극도로 시급한 경우에만 전화 발신 필요)"
    else:
        return f"현재 시각 {time_str} [DAY_SHIFT] (주간 교대 시간대 - 정상 근무 중)"

@tool
def get_next_flight_schedule(unit_id: int) -> str:
    """대상 엔진 유닛의 다음 이륙(비행) 스케줄까지 남은 예상 시간(사이클 수)을 가상 스케줄러 DB에서 조회합니다."""
    remaining_cycles = _get_mock_flight_schedule(unit_id)
    return f"엔진 #{unit_id} 다음 비행 예정: {remaining_cycles} 사이클 남음."

@tool
def get_fleet_contacts(unit_id: int) -> str:
    """대상 엔진이 속한 정비구역(Zone)과 담당 정비 책임자의 성함, 휴대폰 번호 및 슬랙 채널을 조회합니다."""
    contact = _get_mock_contact(unit_id)
    return str(contact)

@tool
def trigger_emergency_slack(unit_id: int, slack_channel: str, message: str) -> str:
    """근무 중인 정비조 전체가 볼 수 있도록 공식 메신저 슬랙(Slack) 채널로 진단 에러 상황 및 작업지시 대기를 브로드캐스팅합니다."""
    alert_msg = f"[🚨 EMERGENCY SLACK ALERT] [Channel: {slack_channel}] 엔진 #{unit_id} 경보: {message}"
    print(alert_msg)
    _log_alert(alert_msg)
    return "슬랙 비상 알림 발신 성공"

@tool
def trigger_tts_voice_call(unit_id: int, phone_number: str, manager_name: str, message: str) -> str:
    """야간 숙면 중인 담당자 휴대폰 번호로 비상 자동 전화(TTS Voice Call)를 강제 발신하여 정비사를 즉각 깨고 음성 지침을 안내합니다."""
    alert_msg = f"[📞 TTS VOICE CALL INITIATED] [Recipient: {manager_name} ({phone_number})] 지침 전달: {message}"
    print(alert_msg)
    _log_alert(alert_msg)
    return "정비사 비상 유선 전화 자동 연결 및 TTS 지침 안내 완료"

# --- 3. SmartAlertAgent 클래스 구현 ---

class SmartAlertAgent:
    """
    위기 감지 시 현재 시간대(주/야)와 엔진 비행 예정 시간의 시급성을 자율 종합 판단하여
    슬랙 채널 전송 또는 정비사 긴급 전화 호출(TTS Voice Call)을 차등 실행하는 지능형 가드 에이전트.
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.tools = [
            check_current_time_context,
            get_next_flight_schedule,
            get_fleet_contacts,
            trigger_emergency_slack,
            trigger_tts_voice_call
        ]
        
        self.system_prompt = """당신은 항공 제트엔진 MRO 관제소의 지능형 알람 필터링 에이전트(SmartAlertAgent)입니다.
당신의 주 업무는 엔진 이상 징후 감지 시, 정비사들의 알람 피로(Alert Fatigue)를 막고 적절한 조치를 취하도록 다음 규칙에 따라 알람 수위와 채널을 결정하는 것입니다.

[알람 선택 규칙]
1. `check_current_time_context` 도구로 야간(NIGHT_OFF_HOURS)인지 주간(DAY_SHIFT)인지 판별하십시오.
2. `get_next_flight_schedule` 도구로 정비 시점까지 남은 비행 스케줄(사이클) 여유를 확인하십시오.
3. `get_fleet_contacts` 도구로 수신할 담당 정비사의 연락처를 확인하십시오.
4. 아래에 따라 최종 도구를 발신하십시오:
   - [규칙 1: 주간(DAY_SHIFT)인 경우]
     정비사들이 근무 중이므로, 슬랙 공식 알람 도구(`trigger_emergency_slack`)만 발신하여 교대 정비원 전체가 공유하도록 하십시오.
   - [규칙 2: 야간(NIGHT_OFF_HOURS)이고 다음 비행까지 여유(5 사이클 초과)가 있는 경우]
     정비사가 자고 있으므로 슬랙 공식 알람 도구(`trigger_emergency_slack`)만 전송해 기록을 남기고, 밤샘 전화를 걸어 깨우지 마십시오.
   - [규칙 3: 야간(NIGHT_OFF_HOURS)이고 다음 비행이 임박(5 사이클 이하)한 긴급 상황인 경우]
     매우 긴박합니다. 자고 있는 정비사를 즉각 깨워야 하므로 자동 TTS 비상 전화 도구(`trigger_tts_voice_call`)를 호출하십시오.

도구 실행 결과를 얻은 후, 어떤 생각(Thought) 과정을 거쳐 최종 알람을 실행했는지 간략하고 전문적인 한글 답변으로 요약하여 종료해 주십시오."""

        self.model_with_tools = None
        if self.api_key:
            try:
                # 최신 langchain-openai의 bind_tools를 사용하는 자율 에이전트 루프 초기화
                chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=self.api_key)
                self.model_with_tools = chat.bind_tools(self.tools)
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain ChatOpenAI: {e}. Running in Static Fallback Mode.")

    def run_alert_logic(self, unit_id: int, tick: int, status: str, rul: float) -> str:
        """
        위기 상황을 판단하여 알람을 결정합니다.
        API Key가 없거나 에러가 발생한 경우 정적 대체 로직(Static Fallback)으로 작동합니다.
        """
        if not self.model_with_tools:
            return self._run_static_fallback(unit_id, tick, status, rul)
            
        try:
            user_input = f"엔진 #{unit_id}에서 이상 상태 '{status}' 감지 (예측 RUL: {rul:.1f}). 시뮬레이션 틱: {tick}. 최적의 비상 전송 수단을 선별하여 동작시켜 줘."
            
            # ReAct (Reasoning and Action) 루프 직접 구현
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input)
            ]
            
            # 최대 5회 툴 호출 반복
            for i in range(5):
                print(f"[SmartAlertAgent] Thinking Step {i+1}...")
                response = self.model_with_tools.invoke(messages)
                messages.append(response)
                
                # 툴 호출이 없으면 루프를 종료하고 최종 컨텐츠 반환
                if not response.tool_calls:
                    return response.content
                    
                # 툴 실행
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # 매칭되는 툴 찾기
                    matching_tool = next((t for t in self.tools if t.name == tool_name), None)
                    if matching_tool:
                        print(f"[SmartAlertAgent] Executing Tool: {tool_name} with args: {tool_args}")
                        tool_output = matching_tool.invoke(tool_args)
                        messages.append(ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call["id"]
                        ))
                    else:
                        messages.append(ToolMessage(
                            content=f"Error: Tool '{tool_name}' not found.",
                            tool_call_id=tool_call["id"]
                        ))
            return "Error: Maximum agent loop iterations exceeded."
            
        except Exception as exc:
            print(f"Warning: LangChain SmartAlertAgent run failed: {exc}. Falling back to static logic.")
            return self._run_static_fallback(unit_id, tick, status, rul)

    def _run_static_fallback(self, unit_id: int, tick: int, status: str, rul: float) -> str:
        """정적 Fallback 규칙 엔진"""
        minutes_in_day = tick % 1440
        is_night = minutes_in_day >= 1320 or minutes_in_day < 480
        remaining_cycles = _get_mock_flight_schedule(unit_id)
        contact = _get_mock_contact(unit_id)
        
        message = f"유닛 #{unit_id} 상태 '{status}' 이탈 (RUL: {rul:.1f})"
        
        if is_night and remaining_cycles <= 5:
            trigger_tts_voice_call.invoke({
                "unit_id": unit_id,
                "phone_number": contact["phone"],
                "manager_name": contact["manager"],
                "message": message
            })
            return f"[Fallback] 야간 초긴급 전화 호출 완료 (대상: {contact['manager']})"
        else:
            trigger_emergency_slack.invoke({
                "unit_id": unit_id,
                "slack_channel": contact["slack_channel"],
                "message": message
            })
            return f"[Fallback] 업무 공식 슬랙 전송 완료 (채널: {contact['slack_channel']})"
